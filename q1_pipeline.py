"""Run the full Q1 pipeline from clean match data to exported predictions.

The pipeline has three jobs:

1. turn cleaned match history into team and form features;
2. predict home and away corner means for every target match;
3. calibrate conditional variance and market-aware adjustments.
   Raw market odds are not used as target values or copied directly into the prediction. 
   Market information is transformed into higher-level signals, 
   like implied total level, side strength, certainty, and model–market disagreement, 
   and used only to adjust the predicted distribution width and calibration.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import nbinom, poisson, skellam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from team_strength import walk_matches
from team_variance import fit_dispersion


matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# Per-team rolling form features built in the one-row-per-team-per-match table.
# Naming convention:
# - corners_for: the team's own corner count in that match
# - corners_against: the opponent's corner count against that team
# - goals_for / goals_against: the team's own goals scored / conceded
# - goal_difference: goals_for minus goals_against
# - ewm_half_life_5: exponentially weighted moving average with half-life 5 matches
# - rolling_window_20: rolling summary over the last 20 matches
TEAM_FEATURE_COLS = [
    "corners_for_ewm_half_life_5",
    "corners_against_ewm_half_life_5",
    "goals_for_rolling_window_20",
    "goals_against_rolling_window_20",
    "goal_difference_rolling_window_20",
    "corners_for_std_rolling_window_20",
    "corners_against_std_rolling_window_20",
]

# sub-model: team strength features built to predict the corner counts
# As team strength is the main signal for corner counts （verified by feature importance）
# These summarize each team's attack and defence level, both overall and in
# home/away-specific contexts, plus the implied corner-rate balance.
STAGE1A_COLS = [
    "home_attack_rating",
    "home_defence_leakiness",
    "away_attack_rating",
    "away_defence_leakiness",
    "log_lambda_home",
    "log_lambda_away",
    "strength_diff",
    "home_attack_at_home",
    "home_defence_at_home",
    "away_attack_at_away",
    "away_defence_at_away",
    "venue_strength_diff",
]

# LightGBM model using the output of the team strength model plus some raw features. 
# 28-feature subset survived the feature-ablation check. 
FEATURE_COLS = [
    "season_id",
    "gameweek",
    "home_attack_rating",
    "home_defence_leakiness",
    "away_attack_rating",
    "away_defence_leakiness",
    "home_attack_at_home",
    "home_defence_at_home",
    "away_attack_at_away",
    "away_defence_at_away",
    "log_lambda_home",
    "log_lambda_away",
    "strength_diff",
    "venue_strength_diff",
    "league_log_baseline_away",
    "home_corners_for_ewm_half_life_5",
    "home_corners_against_ewm_half_life_5",
    "away_corners_for_ewm_half_life_5",
    "away_corners_against_ewm_half_life_5",
    "home_goals_against_rolling_window_20",
    "home_goal_difference_rolling_window_20",
    "away_goals_for_rolling_window_20",
    "away_goals_against_rolling_window_20",
    "away_goal_difference_rolling_window_20",
    "home_corners_for_std_rolling_window_20",
    "home_corners_against_std_rolling_window_20",
    "away_corners_for_std_rolling_window_20",
    "away_corners_against_std_rolling_window_20",
]


CATEGORICAL_COLS = ["season_id"]

# Fallback probability when only one side of a two-way market is present.
MARKET_TWO_WAY_FILL = 1 / 3

# Lower and upper bounds for total-corner means.
MARKET_TARGET_MIN_TOTAL = 0.1
MARKET_TARGET_MAX_TOTAL = 30.0

# Quantile predictions thresholds for corners
RESIDUAL_QUANTILES = (0.10, 0.50, 0.90)

# Residual-quantile bucket number, which controls the granularity of the variance. 
# More buckets can capture finer patterns but risk overfitting and sparsity.
RESIDUAL_QUANTILE_BUCKETS = 4

# Minimum sample size before a residual bucket is trusted.
RESIDUAL_QUANTILE_MIN_GROUP = 12

# Width of a standard-normal 10th-to-90th percentile interval.
# This turns a q10/q90 spread into an approximate variance estimate.
NORMAL_Q10_Q90_SPAN = 2.5631031310892007


@dataclass
class PipelineArtifacts:
    model_data: pd.DataFrame
    betting_matches: pd.DataFrame
    features: pd.DataFrame
    betting_match_features: pd.DataFrame
    train_feat: pd.DataFrame
    val_feat: pd.DataFrame
    results: pd.DataFrame
    gap: pd.DataFrame
    val_predictions: pd.DataFrame
    bet_predictions: pd.DataFrame


def print_heading(title: str) -> None:
    """Print a small console section header."""
    print(f"\n=== {title} ===")


def print_frame(name: str, frame: pd.DataFrame, decimals: int = 3) -> None:
    """Print a small table with a label."""
    print(f"\n{name}")
    print(frame.round(decimals).to_string())


def write_csv_with_fallback(frame: pd.DataFrame, path: Path) -> Path:
    """Write one CSV file. If the file is locked, write a fallback copy."""
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.latest{path.suffix}")
        frame.to_csv(fallback, index=False)
        print(f"Permission denied for {path.name}; wrote {fallback.name} instead.")
        return fallback


def irreducible_mae(poisson_mean: float, n: int = 200_000) -> float:
    """Estimate the best MAE a pure Poisson process could reach."""
    return float(np.mean(np.abs(np.random.poisson(poisson_mean, n) - poisson_mean)))


def latest_snapshot(
    fixtures: pd.DataFrame,
    side: str,
    snapshot: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Attach the latest pre-match team row for one side of each fixture."""
    team_col = f"{side}_team_id"
    out = fixtures[["match_id", "date_time", team_col]].rename(columns={team_col: "team_id"}).copy()
    out = out.sort_values("date_time").reset_index(drop=True)
    out = pd.merge_asof(out, snapshot, on="date_time", by="team_id", direction="backward")
    renamed = {col: f"{side}_{col}" for col in cols}
    return out[["match_id"] + cols].rename(columns=renamed)


def poisson_nll(observed_corners: pd.Series | np.ndarray, predicted_mean: np.ndarray) -> float:
    """Compute average Poisson negative log-likelihood."""
    safe_predicted_mean = np.clip(predicted_mean, 1e-6, None)
    return float(-poisson.logpmf(np.asarray(observed_corners, int), safe_predicted_mean).mean())


def poisson_crps(
    observed_corners: pd.Series | np.ndarray,
    predicted_mean: np.ndarray,
    k_max: int = 40,
) -> float:
    """Approximate Poisson CRPS on a finite count grid."""
    observed_array = np.asarray(observed_corners, int)
    predicted_mean_array = np.asarray(predicted_mean, float)
    corner_grid = np.arange(k_max).reshape(-1, 1)
    poisson_cdf = poisson.cdf(corner_grid, predicted_mean_array.reshape(1, -1))
    realised_tail_indicator = (corner_grid >= observed_array.reshape(1, -1)).astype(float)
    return float(np.mean(np.sum((poisson_cdf - realised_tail_indicator) ** 2, axis=0)))


def devig_two_way(price_over: float, price_under: float) -> tuple[float, float] | None:
    """Turn a 2-way price pair into no-vig probabilities."""
    if pd.isna(price_over) or pd.isna(price_under) or price_over <= 1 or price_under <= 1:
        return None
    inv_over = 1 / price_over
    inv_under = 1 / price_under
    total = inv_over + inv_under
    if total <= 0:
        return None
    return inv_over / total, inv_under / total


def devig_three_way(price_home: float, price_away: float, price_draw: float) -> tuple[float, float, float] | None:
    """Turn a 3-way price triplet into no-vig probabilities."""
    if (
        pd.isna(price_home)
        or pd.isna(price_away)
        or pd.isna(price_draw)
        or price_home <= 1
        or price_away <= 1
        or price_draw <= 1
    ):
        return None
    inv_home = 1 / price_home
    inv_away = 1 / price_away
    inv_draw = 1 / price_draw
    total = inv_home + inv_away + inv_draw
    if total <= 0:
        return None
    return inv_home / total, inv_away / total, inv_draw / total


def solve_total_mean_from_ou(line: float, prob_over: float) -> float | None:
    """Infer the market's total-corner mean from one OU line and price."""
    if pd.isna(line) or pd.isna(prob_over):
        return None
    
    # Only invert clean half-lines such as 8.5, 9.5, 10.5.
    # These avoid push outcomes.
    fractional = round(float(line) % 1, 3)
    if fractional != 0.5:
        return None

    # Example:
    #   line = 9.5
    #   threshold = 9
    #   P(over 9.5) = P(total corners > 9)

    threshold = int(np.floor(line))

    def objective(total_mean: float) -> float:
        # Difference between Poisson-implied over probability and
        # market-implied over probability.
        return float(1 - poisson.cdf(threshold, total_mean) - prob_over)

    try:
        return float(brentq(objective, MARKET_TARGET_MIN_TOTAL, MARKET_TARGET_MAX_TOTAL))
    except ValueError:
        return None


def skellam_probabilities(
    predicted_home_mean: float,
    predicted_away_mean: float,
) -> tuple[float, float, float]:
    """Turn home and away means into home, draw, and away probabilities."""

    prob_draw = float(skellam.pmf(0, predicted_home_mean, predicted_away_mean))
    prob_away = float(skellam.cdf(-1, predicted_home_mean, predicted_away_mean))
    prob_home = float(1 - prob_draw - prob_away)
    return prob_home, prob_away, prob_draw


def solve_home_away_means_from_market(
    total_mean: float,
    prob_home: float,
    prob_away: float,
    prob_draw: float,
) -> tuple[float, float] | None:
    """Split a total mean into home and away means that best match 1X2 prices."""
    if total_mean <= 0:
        return None

    def loss(candidate_home_mean: float) -> float:
        candidate_away_mean = total_mean - candidate_home_mean
        if candidate_home_mean <= 0 or candidate_away_mean <= 0:
            return 1e9
        model_home, model_away, model_draw = skellam_probabilities(candidate_home_mean, candidate_away_mean)
        return float(
            (model_home - prob_home) ** 2
            + (model_away - prob_away) ** 2
            + (model_draw - prob_draw) ** 2
        )

    try:
        fit = minimize_scalar(loss, bounds=(0.05, total_mean - 0.05), method="bounded")
    except ValueError:
        return None
    if not fit.success:
        return None
    predicted_home_mean = float(fit.x)
    predicted_away_mean = float(total_mean - predicted_home_mean)
    if predicted_home_mean <= 0 or predicted_away_mean <= 0:
        return None
    return predicted_home_mean, predicted_away_mean


def infer_market_means_for_row(row: pd.Series) -> tuple[float, float] | None:
    """Infer market-implied home and away means from one price row."""
    devig_ou = devig_two_way(row["ou_over_price"], row["ou_under_price"])
    devig_1x2 = devig_three_way(row["p1x2_home_price"], row["p1x2_away_price"], row["p1x2_draw_price"])
    if devig_ou is None or devig_1x2 is None:
        return None
    prob_over, _ = devig_ou
    prob_home, prob_away, prob_draw = devig_1x2
    total_mean = solve_total_mean_from_ou(row["ou_line"], prob_over)
    if total_mean is None:
        return None
    return solve_home_away_means_from_market(total_mean, prob_home, prob_away, prob_draw)


def compute_market_feature_row(row: pd.Series) -> pd.Series:
    """Turn one row of raw prices into cleaner market features."""
    probs_1x2 = devig_three_way(row.get("p1x2_home_price"), row.get("p1x2_away_price"), row.get("p1x2_draw_price"))
    probs_ou = devig_two_way(row.get("ou_over_price"), row.get("ou_under_price"))
    probs_hc = devig_two_way(row.get("hc_home_price"), row.get("hc_away_price"))

    market_home_mean = row.get("market_target_home", np.nan)
    market_away_mean = row.get("market_target_away", np.nan)
    market_total_mean = np.nan
    market_diff_mean = np.nan
    if pd.notna(market_home_mean) and pd.notna(market_away_mean):
        market_total_mean = float(market_home_mean + market_away_mean)
        market_diff_mean = float(market_home_mean - market_away_mean)

    p_1x2_home = probs_1x2[0] if probs_1x2 is not None else MARKET_TWO_WAY_FILL
    p_1x2_away = probs_1x2[1] if probs_1x2 is not None else MARKET_TWO_WAY_FILL
    p_1x2_draw = probs_1x2[2] if probs_1x2 is not None else MARKET_TWO_WAY_FILL
    p_ou_over = probs_ou[0] if probs_ou is not None else 0.5
    p_ou_under = probs_ou[1] if probs_ou is not None else 0.5
    p_hc_home = probs_hc[0] if probs_hc is not None else 0.5
    p_hc_away = probs_hc[1] if probs_hc is not None else 0.5

    total_certainty = abs(p_ou_over - 0.5)
    side_certainty = abs(p_1x2_home - p_1x2_away)
    handicap_certainty = abs(p_hc_home - 0.5)
    certainty_components = [total_certainty, side_certainty]
    if probs_hc is not None:
        certainty_components.append(handicap_certainty)
    market_quantile_certainty = float(np.mean(certainty_components))

    predicted_home_mean = float(row["predicted_home_mean"])
    predicted_away_mean = float(row["predicted_away_mean"])
    predicted_total_mean = float(row["predicted_total_mean"])
    predicted_diff_mean = float(row["predicted_diff_mean"])

    market_vs_model_home_gap = market_home_mean - predicted_home_mean if pd.notna(market_home_mean) else 0.0
    market_vs_model_away_gap = market_away_mean - predicted_away_mean if pd.notna(market_away_mean) else 0.0
    market_vs_model_total_gap = market_total_mean - predicted_total_mean if pd.notna(market_total_mean) else 0.0
    market_vs_model_diff_gap = market_diff_mean - predicted_diff_mean if pd.notna(market_diff_mean) else 0.0

    return pd.Series(
        {
            "market_prob_1x2_home": p_1x2_home,
            "market_prob_1x2_away": p_1x2_away,
            "market_prob_1x2_draw": p_1x2_draw,
            "market_prob_ou_over": p_ou_over,
            "market_prob_ou_under": p_ou_under,
            "market_prob_hc_home": p_hc_home,
            "market_prob_hc_away": p_hc_away,
            "market_home_mean": market_home_mean,
            "market_away_mean": market_away_mean,
            "market_total_mean": market_total_mean,
            "market_diff_mean": market_diff_mean,
            "market_total_certainty": total_certainty,
            "market_side_certainty": side_certainty,
            "market_handicap_certainty": handicap_certainty,
            "market_quantile_certainty": market_quantile_certainty,
            "market_draw_prob": p_1x2_draw,
            "market_vs_model_home_gap": market_vs_model_home_gap,
            "market_vs_model_away_gap": market_vs_model_away_gap,
            "market_vs_model_total_gap": market_vs_model_total_gap,
            "market_vs_model_diff_gap": market_vs_model_diff_gap,
            "market_tail_width_proxy": total_certainty + 0.5 * market_quantile_certainty,
            "market_upper_tail_home_proxy": max(p_1x2_home - p_1x2_away, 0.0) + max(p_hc_home - 0.5, 0.0),
            "market_upper_tail_away_proxy": max(p_1x2_away - p_1x2_home, 0.0) + max(p_hc_away - 0.5, 0.0),
        }
    )


def quantile_bucket_edges(values: pd.Series, bucket_count: int = RESIDUAL_QUANTILE_BUCKETS) -> np.ndarray:
    """Build quantile bucket edges with open tails."""
    clean_values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if clean_values.empty:
        return np.array([-np.inf, np.inf], dtype=float)
    raw_edges = clean_values.quantile(np.linspace(0.0, 1.0, bucket_count + 1)).to_numpy(dtype=float)
    raw_edges = np.unique(raw_edges)
    if raw_edges.size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    raw_edges[0] = -np.inf
    raw_edges[-1] = np.inf
    return raw_edges.astype(float)


def assign_bucket_codes(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Map continuous values into integer bucket IDs."""
    if len(edges) <= 2:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    bucket_codes = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    return bucket_codes.astype("float").fillna(0).astype(int)


def fit_residual_quantile_lookup(
    calibration_rows: pd.DataFrame,
    prediction_col: str,
    actual_col: str,
    certainty_col: str,
    rolling_std_col: str,
    gap_col: str,
    side_name: str,
) -> dict[str, object]:
    """Fit residual quantile lookup tables on partition A."""
    work = calibration_rows[
        [prediction_col, actual_col, certainty_col, rolling_std_col, gap_col]
    ].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work[certainty_col] = work[certainty_col].fillna(0.0)
    work[rolling_std_col] = work[rolling_std_col].fillna(work[rolling_std_col].median())
    work[rolling_std_col] = work[rolling_std_col].fillna(0.0)
    work[gap_col] = work[gap_col].fillna(0.0)
    work["residual"] = work[actual_col] - work[prediction_col]

    bucket_specs = {
        "predicted_mean_bucket": quantile_bucket_edges(work[prediction_col]),
        "market_certainty_bucket": quantile_bucket_edges(work[certainty_col]),
        "rolling_std_bucket": quantile_bucket_edges(work[rolling_std_col]),
        "market_gap_bucket": quantile_bucket_edges(work[gap_col]),
    }
    for bucket_name, edges in bucket_specs.items():
        source_col = {
            "predicted_mean_bucket": prediction_col,
            "market_certainty_bucket": certainty_col,
            "rolling_std_bucket": rolling_std_col,
            "market_gap_bucket": gap_col,
        }[bucket_name]
        work[bucket_name] = assign_bucket_codes(work[source_col], edges)

    lookup_levels = [
        ("full", ["predicted_mean_bucket", "market_certainty_bucket", "rolling_std_bucket", "market_gap_bucket"]),
        ("no_gap", ["predicted_mean_bucket", "market_certainty_bucket", "rolling_std_bucket"]),
        ("mean_cert", ["predicted_mean_bucket", "market_certainty_bucket"]),
        ("mean_only", ["predicted_mean_bucket"]),
    ]

    lookup_tables: list[dict[str, object]] = []
    for level_name, group_cols in lookup_levels:
        rows: list[tuple[tuple[int, ...], dict[str, float]]] = []
        grouped = work.groupby(group_cols, observed=True)["residual"]
        for key, residuals in grouped:
            if len(residuals) < RESIDUAL_QUANTILE_MIN_GROUP:
                continue
            if not isinstance(key, tuple):
                key = (int(key),)
            rows.append(
                (
                    tuple(int(item) for item in key),
                    {
                        "q10": float(residuals.quantile(RESIDUAL_QUANTILES[0])),
                        "q50": float(residuals.quantile(RESIDUAL_QUANTILES[1])),
                        "q90": float(residuals.quantile(RESIDUAL_QUANTILES[2])),
                        "n": int(len(residuals)),
                    },
                )
            )
        lookup_tables.append({"name": level_name, "group_cols": group_cols, "mapping": dict(rows)})

    global_residual = work["residual"]
    global_quantiles = {
        "q10": float(global_residual.quantile(RESIDUAL_QUANTILES[0])),
        "q50": float(global_residual.quantile(RESIDUAL_QUANTILES[1])),
        "q90": float(global_residual.quantile(RESIDUAL_QUANTILES[2])),
        "n": int(len(global_residual)),
    }
    print(
        f"{side_name} residual quantile lookup fitted on {len(work)} rows | "
        f"levels={[(item['name'], len(item['mapping'])) for item in lookup_tables]}"
    )
    return {
        "prediction_col": prediction_col,
        "certainty_col": certainty_col,
        "rolling_std_col": rolling_std_col,
        "gap_col": gap_col,
        "bucket_edges": bucket_specs,
        "tables": lookup_tables,
        "global": global_quantiles,
    }


def apply_residual_quantile_lookup(
    target_rows: pd.DataFrame,
    lookup: dict[str, object],
) -> pd.DataFrame:
    """Apply one residual quantile lookup table to new rows."""
    work = target_rows.copy()
    prediction_col = str(lookup["prediction_col"])
    certainty_col = str(lookup["certainty_col"])
    rolling_std_col = str(lookup["rolling_std_col"])
    gap_col = str(lookup["gap_col"])

    fill_values = {
        certainty_col: 0.0,
        rolling_std_col: 0.0,
        gap_col: 0.0,
    }
    for source_col, fill_value in fill_values.items():
        work[source_col] = work[source_col].replace([np.inf, -np.inf], np.nan).fillna(fill_value)

    for bucket_name, edges in lookup["bucket_edges"].items():
        source_col = {
            "predicted_mean_bucket": prediction_col,
            "market_certainty_bucket": certainty_col,
            "rolling_std_bucket": rolling_std_col,
            "market_gap_bucket": gap_col,
        }[bucket_name]
        work[bucket_name] = assign_bucket_codes(work[source_col], edges)

    q10_values: list[float] = []
    q50_values: list[float] = []
    q90_values: list[float] = []
    level_names: list[str] = []

    for _, row in work.iterrows():
        quantiles = None
        level_name = "global"
        for table in lookup["tables"]:
            key = tuple(int(row[group_col]) for group_col in table["group_cols"])
            quantiles = table["mapping"].get(key)
            if quantiles is not None:
                level_name = str(table["name"])
                break
        if quantiles is None:
            quantiles = lookup["global"]
        base_prediction = float(row[prediction_col])
        quantile_triplet = np.array(
            [
                max(base_prediction + float(quantiles["q10"]), 0.1),
                max(base_prediction + float(quantiles["q50"]), 0.1),
                max(base_prediction + float(quantiles["q90"]), 0.1),
            ],
            dtype=float,
        )
        quantile_triplet.sort()
        q10_values.append(float(quantile_triplet[0]))
        q50_values.append(float(quantile_triplet[1]))
        q90_values.append(float(quantile_triplet[2]))
        level_names.append(level_name)

    result = pd.DataFrame(index=work.index)
    result["q10"] = q10_values
    result["q50"] = q50_values
    result["q90"] = q90_values
    result["lookup_level"] = level_names
    result["quantile_variance"] = np.maximum(((result["q90"] - result["q10"]) / NORMAL_Q10_Q90_SPAN) ** 2, 1e-6)
    return result


def attach_market_quantile_features(
    betting_match_features: pd.DataFrame,
    predicted_home_bet: np.ndarray,
    predicted_away_bet: np.ndarray,
    global_std_c: float,
) -> pd.DataFrame:
    """Build market-based quantile features and attach them to betting rows."""
    enriched = betting_match_features.copy()
    enriched["predicted_home_mean"] = predicted_home_bet
    enriched["predicted_away_mean"] = predicted_away_bet
    enriched["predicted_total_mean"] = predicted_home_bet + predicted_away_bet
    enriched["predicted_diff_mean"] = predicted_home_bet - predicted_away_bet
    enriched["home_corners_for_std_rolling_window_20"] = enriched["home_corners_for_std_rolling_window_20"].fillna(global_std_c)
    enriched["away_corners_for_std_rolling_window_20"] = enriched["away_corners_for_std_rolling_window_20"].fillna(global_std_c)

    if "market_target_home" not in enriched.columns or "market_target_away" not in enriched.columns:
        inferred = enriched.apply(infer_market_means_for_row, axis=1)
        enriched["market_target_home"] = [item[0] if item is not None else np.nan for item in inferred]
        enriched["market_target_away"] = [item[1] if item is not None else np.nan for item in inferred]

    market_features = enriched.apply(compute_market_feature_row, axis=1)
    enriched = pd.concat([enriched, market_features], axis=1)

    a_mask = enriched["partition"] == "A"
    b_mask = enriched["partition"] == "B"
    partition_a = enriched.loc[a_mask].sort_values("date_time").copy()
    partition_b = enriched.loc[b_mask].sort_values("date_time").copy()
    n_splits = min(4, max(2, len(partition_a) // 120))
    tscv = TimeSeriesSplit(n_splits=n_splits) if len(partition_a) >= 240 else None

    side_specs = [
        {
            "side_name": "home",
            "prediction_col": "predicted_home_mean",
            "actual_col": "home_corners",
            "rolling_std_col": "home_corners_for_std_rolling_window_20",
            "gap_col": "market_vs_model_home_gap",
            "output_cols": ("pred_home_q10", "pred_home_q50", "pred_home_q90", "quantile_sigma2_home", "quantile_lookup_level_home"),
        },
        {
            "side_name": "away",
            "prediction_col": "predicted_away_mean",
            "actual_col": "away_corners",
            "rolling_std_col": "away_corners_for_std_rolling_window_20",
            "gap_col": "market_vs_model_away_gap",
            "output_cols": ("pred_away_q10", "pred_away_q50", "pred_away_q90", "quantile_sigma2_away", "quantile_lookup_level_away"),
        },
    ]

    print_heading("Market Features")
    print(
        "Exported market abstractions:\n"
        "- no-vig probs: market_prob_1x2_*, market_prob_ou_*, market_prob_hc_*\n"
        "- market centers: market_home_mean, market_away_mean, market_total_mean, market_diff_mean\n"
        "- certainty / tails: market_total_certainty, market_side_certainty, market_handicap_certainty, "
        "market_quantile_certainty, market_tail_width_proxy, market_upper_tail_*_proxy\n"
        "- disagreement: market_vs_model_home_gap, market_vs_model_away_gap, market_vs_model_total_gap, market_vs_model_diff_gap"
    )

    for spec in side_specs:
        q10_col, q50_col, q90_col, variance_col, level_col = spec["output_cols"]
        certainty_col = "market_quantile_certainty"
        oof_quantiles = pd.DataFrame(index=partition_a.index, columns=["q10", "q50", "q90", "lookup_level", "quantile_variance"])

        if tscv is not None:
            for train_idx, val_idx in tscv.split(partition_a):
                fold_train = partition_a.iloc[train_idx]
                fold_val = partition_a.iloc[val_idx]
                lookup = fit_residual_quantile_lookup(
                    fold_train,
                    spec["prediction_col"],
                    spec["actual_col"],
                    certainty_col,
                    spec["rolling_std_col"],
                    spec["gap_col"],
                    spec["side_name"],
                )
                fold_quantiles = apply_residual_quantile_lookup(fold_val, lookup)
                oof_quantiles.loc[fold_val.index, :] = fold_quantiles[["q10", "q50", "q90", "lookup_level", "quantile_variance"]].values

        final_lookup = fit_residual_quantile_lookup(
            partition_a,
            spec["prediction_col"],
            spec["actual_col"],
            certainty_col,
            spec["rolling_std_col"],
            spec["gap_col"],
            spec["side_name"],
        )
        fill_quantiles_a = apply_residual_quantile_lookup(partition_a, final_lookup)
        oof_quantiles = oof_quantiles.where(oof_quantiles.notna(), fill_quantiles_a)
        holdout_quantiles_b = apply_residual_quantile_lookup(partition_b, final_lookup)

        enriched.loc[partition_a.index, q10_col] = oof_quantiles["q10"].to_numpy(dtype=float)
        enriched.loc[partition_a.index, q50_col] = oof_quantiles["q50"].to_numpy(dtype=float)
        enriched.loc[partition_a.index, q90_col] = oof_quantiles["q90"].to_numpy(dtype=float)
        enriched.loc[partition_a.index, variance_col] = oof_quantiles["quantile_variance"].to_numpy(dtype=float)
        enriched.loc[partition_a.index, level_col] = oof_quantiles["lookup_level"].astype(str).values

        enriched.loc[partition_b.index, q10_col] = holdout_quantiles_b["q10"].to_numpy(dtype=float)
        enriched.loc[partition_b.index, q50_col] = holdout_quantiles_b["q50"].to_numpy(dtype=float)
        enriched.loc[partition_b.index, q90_col] = holdout_quantiles_b["q90"].to_numpy(dtype=float)
        enriched.loc[partition_b.index, variance_col] = holdout_quantiles_b["quantile_variance"].to_numpy(dtype=float)
        enriched.loc[partition_b.index, level_col] = holdout_quantiles_b["lookup_level"].astype(str).values

        actual_a = partition_a[spec["actual_col"]].to_numpy(dtype=float)
        base_a = partition_a[spec["prediction_col"]].to_numpy(dtype=float)
        actual_b = partition_b[spec["actual_col"]].to_numpy(dtype=float)
        base_b = partition_b[spec["prediction_col"]].to_numpy(dtype=float)
        q50_a = enriched.loc[partition_a.index, q50_col].to_numpy(dtype=float)
        q50_b = enriched.loc[partition_b.index, q50_col].to_numpy(dtype=float)
        q10_b = enriched.loc[partition_b.index, q10_col].to_numpy(dtype=float)
        q90_b = enriched.loc[partition_b.index, q90_col].to_numpy(dtype=float)
        coverage_b = float(np.mean((actual_b >= q10_b) & (actual_b <= q90_b))) if len(actual_b) else np.nan
        print(
            f"{spec['side_name']} quantile diagnostics | "
            f"A MAE base={mean_absolute_error(actual_a, base_a):.4f} q50={mean_absolute_error(actual_a, q50_a):.4f} | "
            f"B MAE base={mean_absolute_error(actual_b, base_b):.4f} q50={mean_absolute_error(actual_b, q50_b):.4f} | "
            f"B 10-90 coverage={coverage_b:.3f}"
        )

    return enriched


def fit_market_teacher(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> LGBMRegressor:
    """Fit a teacher model that copies the market-implied corner means."""
    model = LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    fit_kwargs: dict[str, object] = {}
    if x_val is not None and y_val is not None and len(x_val) > 0:
        fit_kwargs["eval_set"] = [(x_val, y_val)]
        fit_kwargs["callbacks"] = [early_stopping(75, verbose=False)]
    model.fit(x_train, y_train, categorical_feature=CATEGORICAL_COLS, **fit_kwargs)
    return model


def optimise_blend_weight(base_pred: np.ndarray, teacher_pred: np.ndarray, observed: np.ndarray) -> float:
    """Choose the blend weight that gives the lowest MAE on a small grid."""
    grid = np.linspace(0.0, 1.0, 21)
    best_weight = 0.0
    best_mae = np.inf
    for weight in grid:
        blended = (1 - weight) * base_pred + weight * teacher_pred
        mae = mean_absolute_error(observed, blended)
        if mae < best_mae:
            best_mae = mae
            best_weight = float(weight)
    return best_weight


def evaluate_predictions(
    model_name: str,
    actual_home_corners: pd.Series,
    actual_away_corners: pd.Series,
    predicted_home_corners: np.ndarray,
    predicted_away_corners: np.ndarray,
) -> dict[str, float | str]:
    """Compute the main validation metrics."""
    actual_corner_diff = actual_home_corners - actual_away_corners
    predicted_corner_diff = predicted_home_corners - predicted_away_corners
    return {
        "model": model_name,
        "home_MAE": mean_absolute_error(actual_home_corners, predicted_home_corners),
        "away_MAE": mean_absolute_error(actual_away_corners, predicted_away_corners),
        "diff_MAE": mean_absolute_error(actual_corner_diff, predicted_corner_diff),
        "home_RMSE": np.sqrt(mean_squared_error(actual_home_corners, predicted_home_corners)),
        "away_RMSE": np.sqrt(mean_squared_error(actual_away_corners, predicted_away_corners)),
        "diff_RMSE": np.sqrt(mean_squared_error(actual_corner_diff, predicted_corner_diff)),
        "home_NLL": poisson_nll(actual_home_corners, predicted_home_corners),
        "away_NLL": poisson_nll(actual_away_corners, predicted_away_corners),
        "home_CRPS": poisson_crps(actual_home_corners, predicted_home_corners),
        "away_CRPS": poisson_crps(actual_away_corners, predicted_away_corners),
    }


def load_inputs(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the clean input tables and align training to the betting leagues."""
    train = pd.read_parquet(data_dir / "train.parquet")
    betting = pd.read_parquet(data_dir / "betting.parquet")
    all_matches = pd.read_parquet(data_dir / "all_matches.parquet")

    train["date_time"] = pd.to_datetime(train["date_time"])
    betting["date_time"] = pd.to_datetime(betting["date_time"])
    all_matches["date_time"] = pd.to_datetime(all_matches["date_time"])

    betting_matches = betting.drop_duplicates("match_id").copy()
    model_data = train[train["competition_id"].isin(betting_matches["competition_id"])].copy()
    model_data = model_data.sort_values("date_time").reset_index(drop=True)
    model_data["total_corners"] = model_data["home_corners"] + model_data["away_corners"]

    print_heading("Load Data")
    print(
        f"model_data: {model_data.shape} | betting_matches: {betting_matches.shape}\n"
        f"train date range: {model_data['date_time'].min().date()} to {model_data['date_time'].max().date()}\n"
        f"betting date range: {betting_matches['date_time'].min().date()} to {betting_matches['date_time'].max().date()}"
    )
    return train, betting, all_matches, model_data, betting_matches


def baseline_diagnostics(model_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Print basic dispersion stats and the Poisson error floor."""
    targets = ["home_corners", "away_corners", "total_corners"]
    disp = pd.DataFrame(
        {
            "mean": model_data[targets].mean(),
            "variance": model_data[targets].var(),
        }
    )
    disp["variance / mean"] = disp["variance"] / disp["mean"]

    np.random.seed(0)
    bound = pd.DataFrame(
        [
            {
                "target": target,
                "lambda": disp.loc[target, "mean"],
                "sqrt(2*lambda/pi)": float(np.sqrt(2 * disp.loc[target, "mean"] / np.pi)),
                "simulated_MAE": irreducible_mae(float(disp.loc[target, "mean"])),
            }
            for target in targets
        ]
    )

    print_heading("Baseline Diagnostics")
    print_frame("dispersion table", disp)
    print_frame("irreducible floor", bound)
    return disp, bound


def build_strength_features(
    all_matches: pd.DataFrame,
    model_data: pd.DataFrame,
    betting_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Build leak-free team-strength features for train and betting matches."""
    target_match_ids = set(model_data["match_id"].astype(int)) | set(betting_matches["match_id"].astype(int))
    strength_features = walk_matches(all_matches, target_match_ids)

    print_heading("Stage 1a")
    print(
        f"Stage 1a strength features: {strength_features.shape}\n"
        f"update pool: {len(all_matches):,} matches | target pool: {len(target_match_ids):,} matches\n"
        f"home_attack_rating std={strength_features.home_attack_rating.std():.3f} "
        f"range=[{strength_features.home_attack_rating.min():.2f},{strength_features.home_attack_rating.max():.2f}]\n"
        f"strength_diff std={strength_features.strength_diff.std():.3f} "
        f"venue_strength_diff std={strength_features.venue_strength_diff.std():.3f}"
    )
    return strength_features


def build_team_games(model_data: pd.DataFrame) -> pd.DataFrame:
    """Turn match rows into one row per team per match."""
    home_rows = model_data[
        [
            "match_id",
            "date_time",
            "competition_id",
            "home_team_id",
            "home_corners",
            "away_corners",
            "home_ft_score",
            "away_ft_score",
        ]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_corners": "corners_for",
            "away_corners": "corners_against",
            "home_ft_score": "goals_for",
            "away_ft_score": "goals_against",
        }
    )
    home_rows["is_home"] = 1

    away_rows = model_data[
        [
            "match_id",
            "date_time",
            "competition_id",
            "away_team_id",
            "away_corners",
            "home_corners",
            "away_ft_score",
            "home_ft_score",
        ]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_corners": "corners_for",
            "home_corners": "corners_against",
            "away_ft_score": "goals_for",
            "home_ft_score": "goals_against",
        }
    )
    away_rows["is_home"] = 0

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games = team_games.sort_values(["team_id", "date_time", "match_id"]).reset_index(drop=True)

    print_heading("Per-Team Match Table")
    print(
        f"match-level rows: {len(model_data):,} -> team-game rows: {len(team_games):,} (= 2 x match rows)"
    )
    print(team_games.head(4).to_string(index=False))
    return team_games


def add_rolling_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add rolling corner and goal features to the team table."""
    team_games = team_games.copy()
    team_games["corners_for_ewm_half_life_5"] = team_games.groupby("team_id")["corners_for"].transform(
        lambda series: series.shift(1).ewm(halflife=5, ignore_na=True).mean()
    )
    team_games["corners_against_ewm_half_life_5"] = team_games.groupby("team_id")["corners_against"].transform(
        lambda series: series.shift(1).ewm(halflife=5, ignore_na=True).mean()
    )
    team_games["goals_for_rolling_window_20"] = team_games.groupby("team_id")["goals_for"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=3).mean()
    )
    team_games["goals_against_rolling_window_20"] = team_games.groupby("team_id")["goals_against"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=3).mean()
    )
    team_games["goal_difference_rolling_window_20"] = (
        team_games["goals_for_rolling_window_20"] - team_games["goals_against_rolling_window_20"]
    )
    team_games["corners_for_std_rolling_window_20"] = team_games.groupby("team_id")["corners_for"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=5).std()
    )
    team_games["corners_against_std_rolling_window_20"] = team_games.groupby("team_id")["corners_against"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=5).std()
    )
    team_games["n_prior"] = team_games.groupby("team_id").cumcount()

    nan_summary = team_games[
        [
            "corners_for_ewm_half_life_5",
            "corners_against_ewm_half_life_5",
            "goals_for_rolling_window_20",
            "goals_against_rolling_window_20",
            "corners_for_std_rolling_window_20",
            "corners_against_std_rolling_window_20",
        ]
    ].isna().sum()
    print_heading("Stage 1b")
    print(f"team_games with rolling features: {team_games.shape}")
    print_frame("NaN per new feature", nan_summary.to_frame("count"))
    return team_games


def build_match_features(
    model_data: pd.DataFrame,
    betting: pd.DataFrame,
    betting_matches: pd.DataFrame,
    team_games: pd.DataFrame,
    strength_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the final match-level feature tables."""
    home_feats = team_games[team_games["is_home"] == 1][["match_id"] + TEAM_FEATURE_COLS].rename(
        columns={col: f"home_{col}" for col in TEAM_FEATURE_COLS}
    )
    away_feats = team_games[team_games["is_home"] == 0][["match_id"] + TEAM_FEATURE_COLS].rename(
        columns={col: f"away_{col}" for col in TEAM_FEATURE_COLS}
    )

    features = (
        model_data.merge(home_feats, on="match_id", how="left")
        .merge(away_feats, on="match_id", how="left")
        .merge(strength_features, on="match_id", how="left")
    )

    team_snap = team_games[["team_id", "date_time"] + TEAM_FEATURE_COLS].sort_values(
        ["date_time", "team_id"]
    )
    team_snap = team_snap.reset_index(drop=True)
    bet_h = latest_snapshot(betting_matches, "home", team_snap, TEAM_FEATURE_COLS)
    bet_a = latest_snapshot(betting_matches, "away", team_snap, TEAM_FEATURE_COLS)

    x12 = betting[betting["odds_type"] == "1X2"].drop_duplicates("match_id")[
        ["match_id", "oh", "oa", "od"]
    ].copy()
    x12 = x12.rename(
        columns={
            "oh": "p1x2_home_price",
            "oa": "p1x2_away_price",
            "od": "p1x2_draw_price",
        }
    )
    inv_sum = 1 / x12["p1x2_home_price"] + 1 / x12["p1x2_away_price"] + 1 / x12["p1x2_draw_price"]
    x12["p_h_1x2"] = (1 / x12["p1x2_home_price"]) / inv_sum
    x12["p_a_1x2"] = (1 / x12["p1x2_away_price"]) / inv_sum
    x12["p_d_1x2"] = (1 / x12["p1x2_draw_price"]) / inv_sum

    ou = betting[betting["odds_type"] == "OU"].drop_duplicates("match_id")[["match_id", "oh", "oa", "od"]].copy()
    ou = ou.rename(
        columns={
            "oh": "ou_over_price",
            "oa": "ou_under_price",
            "od": "ou_line",
        }
    )
    hc = betting[betting["odds_type"] == "HC"].drop_duplicates("match_id")[["match_id", "oh", "oa", "od"]].copy()
    hc = hc.rename(
        columns={
            "oh": "hc_home_price",
            "oa": "hc_away_price",
            "od": "hc_line",
        }
    )

    betting_match_features = (
        betting_matches.merge(bet_h, on="match_id", how="left")
        .merge(bet_a, on="match_id", how="left")
        .merge(strength_features, on="match_id", how="left")
        .merge(x12, on="match_id", how="left")
        .merge(ou, on="match_id", how="left")
        .merge(hc, on="match_id", how="left")
        .sort_values(["date_time", "competition_id"])
        .reset_index(drop=True)
    )

    for col in ["p_h_1x2", "p_a_1x2", "p_d_1x2"]:
        betting_match_features[col] = betting_match_features[col].fillna(MARKET_TWO_WAY_FILL)

    print_heading("Match-Level Features")
    print(f"features: {features.shape}")
    print(f"betting_match_features: {betting_match_features.shape}")
    print(
        "1X2 coverage: "
        f"{betting_match_features[['p_h_1x2', 'p_a_1x2', 'p_d_1x2']].notna().all(axis=1).mean():.1f}"
    )
    return features, betting_match_features


def fill_missing_features(
    model_data: pd.DataFrame,
    features: pd.DataFrame,
    betting_match_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Fill missing engineered features and report what changed."""
    meta_cols = {
        "match_id",
        "date_time",
        "competition_id",
        "season_id",
        "gameweek",
        "iw",
        "home_team_id",
        "away_team_id",
        "home_corners",
        "away_corners",
        "total_corners",
        "home_ft_score",
        "away_ft_score",
        "corner_diff_home_minus_away",
        "odds_type",
        "oh",
        "oa",
        "od",
        "p1x2_home_price",
        "p1x2_away_price",
        "p1x2_draw_price",
        "ou_over_price",
        "ou_under_price",
        "ou_line",
        "hc_home_price",
        "hc_away_price",
        "hc_line",
    }

    train_feat_cols = [col for col in features.columns if col not in meta_cols]
    bet_feat_cols = [col for col in betting_match_features.columns if col not in meta_cols]
    nan_train = features[train_feat_cols].isna().sum()
    nan_bet = betting_match_features[bet_feat_cols].isna().sum()

    print_heading("NaN Audit")
    print(
        f"features: NaN cells = {nan_train.sum():,} "
        f"({nan_train.sum() / (len(features) * len(train_feat_cols)):.2%})"
    )
    print(
        f"betting_match_features: NaN cells = {nan_bet.sum():,} "
        f"({nan_bet.sum() / (len(betting_match_features) * len(bet_feat_cols)):.2%})"
    )
    print_frame("top feature NaN columns", nan_train[nan_train > 0].sort_values(ascending=False).head(15).to_frame("count"))
    print_frame("top betting NaN columns", nan_bet[nan_bet > 0].sort_values(ascending=False).head(15).to_frame("count"))

    global_mean_h = float(model_data["home_corners"].mean())
    global_mean_a = float(model_data["away_corners"].mean())
    global_mean_g = float(model_data[["home_ft_score", "away_ft_score"]].mean().mean())
    global_std_c = float(
        np.sqrt(0.5 * (model_data["home_corners"].var() + model_data["away_corners"].var()))
    )
    log_mean_h = float(np.log(global_mean_h))
    log_mean_a = float(np.log(global_mean_a))

    for frame in [features, betting_match_features]:
        frame["home_corners_for_ewm_half_life_5"] = frame["home_corners_for_ewm_half_life_5"].fillna(global_mean_h)
        frame["home_corners_against_ewm_half_life_5"] = frame["home_corners_against_ewm_half_life_5"].fillna(global_mean_a)
        frame["away_corners_for_ewm_half_life_5"] = frame["away_corners_for_ewm_half_life_5"].fillna(global_mean_a)
        frame["away_corners_against_ewm_half_life_5"] = frame["away_corners_against_ewm_half_life_5"].fillna(global_mean_h)

        for col in [
            "home_goals_for_rolling_window_20",
            "home_goals_against_rolling_window_20",
            "away_goals_for_rolling_window_20",
            "away_goals_against_rolling_window_20",
        ]:
            frame[col] = frame[col].fillna(global_mean_g)

        frame["home_goal_difference_rolling_window_20"] = (
            frame["home_goals_for_rolling_window_20"] - frame["home_goals_against_rolling_window_20"]
        )
        frame["away_goal_difference_rolling_window_20"] = (
            frame["away_goals_for_rolling_window_20"] - frame["away_goals_against_rolling_window_20"]
        )

        for col in [
            "home_corners_for_std_rolling_window_20",
            "home_corners_against_std_rolling_window_20",
            "away_corners_for_std_rolling_window_20",
            "away_corners_against_std_rolling_window_20",
        ]:
            frame[col] = frame[col].fillna(global_std_c)

        frame["league_log_baseline_home"] = frame["league_log_baseline_home"].fillna(log_mean_h)
        frame["league_log_baseline_away"] = frame["league_log_baseline_away"].fillna(log_mean_a)
        for col in STAGE1A_COLS:
            frame[col] = frame[col].fillna(0.0)
        if "prior_match_count_home" in frame.columns:
            frame["prior_match_count_home"] = frame["prior_match_count_home"].fillna(0)
            frame["prior_match_count_away"] = frame["prior_match_count_away"].fillna(0)

    print_heading("NaN Fill")
    print(f"NaN after fill -- features: {features[train_feat_cols].isna().sum().sum()}")
    print(f"NaN after fill -- betting_match_features: {betting_match_features[bet_feat_cols].isna().sum().sum()}")
    return features, betting_match_features, global_std_c


def split_train_val(
    features: pd.DataFrame,
    betting_match_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the match table into time-ordered train and validation sets."""
    cutoff = features["date_time"].quantile(0.8)
    train_feat = features[features["date_time"] < cutoff].copy()
    val_feat = features[features["date_time"] >= cutoff].copy()

    all_cat = pd.concat(
        [
            train_feat[CATEGORICAL_COLS],
            val_feat[CATEGORICAL_COLS],
            betting_match_features[CATEGORICAL_COLS],
        ],
        axis=0,
    ).astype("category")
    cat_dtype = {col: all_cat[col].dtype for col in CATEGORICAL_COLS}
    for frame in [train_feat, val_feat, betting_match_features]:
        for col in CATEGORICAL_COLS:
            frame[col] = frame[col].astype(cat_dtype[col])

    print_heading("Train / Validation Split")
    split_summary = pd.DataFrame(
        {
            "set": ["train", "val", "betting (holdout)"],
            "n_matches": [len(train_feat), len(val_feat), len(betting_match_features)],
            "date_min": [
                train_feat["date_time"].min().date(),
                val_feat["date_time"].min().date(),
                betting_match_features["date_time"].min().date(),
            ],
            "date_max": [
                train_feat["date_time"].max().date(),
                val_feat["date_time"].max().date(),
                betting_match_features["date_time"].max().date(),
            ],
        }
    )
    print(split_summary.to_string(index=False))
    print(f"\n{len(FEATURE_COLS)} features | val cutoff: {cutoff.date()}")
    return train_feat, val_feat, betting_match_features


def fit_mean_models(
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
) -> tuple[LGBMRegressor, LGBMRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit the home and away corner mean models."""
    train_features = train_feat[FEATURE_COLS]
    val_features = val_feat[FEATURE_COLS]
    actual_home_train = train_feat["home_corners"]
    actual_away_train = train_feat["away_corners"]
    actual_home_val = val_feat["home_corners"]
    actual_away_val = val_feat["away_corners"]
    actual_diff_val = actual_home_val - actual_away_val
    sample_weight = train_feat["iw"].fillna(1) if "iw" in train_feat.columns else None

    lgb_kw = dict(
        objective="regression",
        metric="rmse",
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )

    def fit_single(train_target: pd.Series, validation_target: pd.Series) -> LGBMRegressor:
        model = LGBMRegressor(**lgb_kw)
        model.fit(
            train_features,
            train_target,
            sample_weight=sample_weight,
            eval_set=[(val_features, validation_target)],
            callbacks=[early_stopping(150, verbose=False)],
            categorical_feature=CATEGORICAL_COLS,
        )
        return model

    home_model = fit_single(actual_home_train, actual_home_val)
    away_model = fit_single(actual_away_train, actual_away_val)
    predicted_home_val = np.maximum(home_model.predict(val_features), 0.1)
    predicted_away_val = np.maximum(away_model.predict(val_features), 0.1)
    predicted_home_train = np.maximum(home_model.predict(train_features), 0.1)
    predicted_away_train = np.maximum(away_model.predict(train_features), 0.1)

    print_heading("Stage 2")
    print(
        f"lgb best_iter home={home_model.best_iteration_} away={away_model.best_iteration_}\n"
        f"pred std home={np.std(predicted_home_val):.2f} away={np.std(predicted_away_val):.2f} "
        f"diff={np.std(predicted_home_val - predicted_away_val):.2f}\n"
        f"pred range home=[{predicted_home_val.min():.2f},{predicted_home_val.max():.2f}] "
        f"away=[{predicted_away_val.min():.2f},{predicted_away_val.max():.2f}] "
        f"diff=[{(predicted_home_val - predicted_away_val).min():.2f},{(predicted_home_val - predicted_away_val).max():.2f}]\n"
        f"train_MAE home={mean_absolute_error(actual_home_train, predicted_home_train):.4f} "
        f"away={mean_absolute_error(actual_away_train, predicted_away_train):.4f}\n"
        f"val_MAE home={mean_absolute_error(actual_home_val, predicted_home_val):.4f} "
        f"away={mean_absolute_error(actual_away_val, predicted_away_val):.4f} "
        f"diff={mean_absolute_error(actual_diff_val, predicted_home_val - predicted_away_val):.4f}"
    )

    baseline_home_val = np.full(len(val_feat), actual_home_train.mean())
    baseline_away_val = np.full(len(val_feat), actual_away_train.mean())
    results = pd.DataFrame(
        [
            evaluate_predictions("Mean baseline", actual_home_val, actual_away_val, baseline_home_val, baseline_away_val),
            evaluate_predictions("LGB", actual_home_val, actual_away_val, predicted_home_val, predicted_away_val),
        ]
    ).sort_values("home_MAE").reset_index(drop=True)
    print_frame("validation comparison", results, decimals=4)
    return home_model, away_model, predicted_home_val, predicted_away_val, predicted_home_train, predicted_away_train, results


def distance_to_floor(bound: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Compare model MAE with the baseline and Poisson floor."""
    floor = bound.set_index("target")["simulated_MAE"]
    baseline = results[results["model"] == "Mean baseline"].iloc[0]
    best = results[results["model"] == "LGB"].iloc[0]
    gap = pd.DataFrame(
        {
            "mean_baseline_MAE": [baseline["home_MAE"], baseline["away_MAE"]],
            "irreducible_floor": [floor["home_corners"], floor["away_corners"]],
            f"{best['model']}_MAE": [best["home_MAE"], best["away_MAE"]],
        },
        index=["home", "away"],
    )
    gap["improvable_gap"] = gap["mean_baseline_MAE"] - gap["irreducible_floor"]
    gap["captured_pct"] = 100 * (gap["mean_baseline_MAE"] - gap[f"{best['model']}_MAE"]) / gap["improvable_gap"]
    print_heading("Distance To Poisson Floor")
    print(gap.round(3).to_string())
    return gap


def stage3_calibration(
    betting: pd.DataFrame,
    betting_match_features: pd.DataFrame,
    val_feat: pd.DataFrame,
    model_data: pd.DataFrame,
    home_model: LGBMRegressor,
    away_model: LGBMRegressor,
    predicted_home_val: np.ndarray,
    predicted_away_val: np.ndarray,
    actual_home_val: pd.Series,
    actual_away_val: pd.Series,
    global_std_c: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Fit the variance heads and apply the main holdout corrections."""
    print_heading("Stage 3")
    for side_name, predicted_corners, actual_corners in [
        ("home", predicted_home_val, actual_home_val),
        ("away", predicted_away_val, actual_away_val),
    ]:
        average_corners = float(actual_corners.mean())
        ceiling = float(np.sqrt(max(actual_corners.var() - average_corners, 0.01)))
        pred_std = float(np.std(predicted_corners))
        print(
            f"{side_name:6s} actual_std={float(actual_corners.std()):.2f} "
            f"ceiling={ceiling:.2f} pred_std={pred_std:.2f} captured={100 * pred_std / ceiling:.0f}%"
        )

    bet_sorted = betting_match_features.sort_values("date_time").reset_index(drop=True)
    n_a = int(len(bet_sorted) * 0.40)
    bet_sorted["partition"] = ["A"] * n_a + ["B"] * (len(bet_sorted) - n_a)
    betting_match_features = bet_sorted

    betting_features = betting_match_features[FEATURE_COLS]
    predicted_home_bet = np.maximum(home_model.predict(betting_features), 0.1)
    predicted_away_bet = np.maximum(away_model.predict(betting_features), 0.1)

    a_mask = (betting_match_features["partition"] == "A").values
    actual_home_partition_a = betting_match_features.loc[a_mask, "home_corners"].astype(float).values
    actual_away_partition_a = betting_match_features.loc[a_mask, "away_corners"].astype(float).values
    predicted_home_partition_a = predicted_home_bet[a_mask]
    predicted_away_partition_a = predicted_away_bet[a_mask]
    market_home_prob_a = betting_match_features.loc[a_mask, "p_h_1x2"].fillna(MARKET_TWO_WAY_FILL).values
    market_away_prob_a = betting_match_features.loc[a_mask, "p_a_1x2"].fillna(MARKET_TWO_WAY_FILL).values
    rolling_home_std_a = betting_match_features.loc[a_mask, "home_corners_for_std_rolling_window_20"].fillna(global_std_c).values
    rolling_away_std_a = betting_match_features.loc[a_mask, "away_corners_for_std_rolling_window_20"].fillna(global_std_c).values
    market_certainty_home_a = np.abs(market_home_prob_a - 0.5)
    market_certainty_away_a = np.abs(market_away_prob_a - 0.5)

    dispersion_home = fit_dispersion(
        predicted_home_partition_a,
        actual_home_partition_a,
        market_certainty_home_a,
        rolling_home_std_a,
    )
    dispersion_away = fit_dispersion(
        predicted_away_partition_a,
        actual_away_partition_a,
        market_certainty_away_a,
        rolling_away_std_a,
    )

    print(f"partition A: n={a_mask.sum()} date_max={betting_match_features.loc[a_mask, 'date_time'].max().date()}")
    print(f"partition B: n={(~a_mask).sum()} date_min={betting_match_features.loc[~a_mask, 'date_time'].min().date()}")
    print(
        f"dispersion_home intercept={dispersion_home.intercept:.3f} "
        f"beta(mu)={dispersion_home.coefficients[0]:.3f} "
        f"beta(market_cert)={dispersion_home.coefficients[1]:.3f} "
        f"beta(rolling_std)={dispersion_home.coefficients[2]:.3f} "
        f"calibration_scale={dispersion_home.calibration_scale:.3f}"
    )
    print(
        f"dispersion_away intercept={dispersion_away.intercept:.3f} "
        f"beta(mu)={dispersion_away.coefficients[0]:.3f} "
        f"beta(market_cert)={dispersion_away.coefficients[1]:.3f} "
        f"beta(rolling_std)={dispersion_away.coefficients[2]:.3f} "
        f"calibration_scale={dispersion_away.calibration_scale:.3f}"
    )

    a_resid = pd.DataFrame(
        {
            "competition_id": betting_match_features.loc[a_mask, "competition_id"].values,
            "residual_h": actual_home_partition_a - predicted_home_partition_a,
            "residual_a": actual_away_partition_a - predicted_away_partition_a,
        }
    )
    bias = a_resid.groupby("competition_id")[["residual_h", "residual_a"]].mean()
    b_competitions = betting_match_features.loc[~a_mask, "competition_id"]
    correction_h = np.clip(b_competitions.map(bias["residual_h"]).fillna(0).values, -2, 2)
    correction_a = np.clip(b_competitions.map(bias["residual_a"]).fillna(0).values, -2, 2)

    adjusted_home_bet = predicted_home_bet.copy()
    adjusted_away_bet = predicted_away_bet.copy()
    adjusted_home_bet[~a_mask] += correction_h
    adjusted_away_bet[~a_mask] += correction_a
    adjusted_home_bet = np.maximum(adjusted_home_bet, 0.1)
    adjusted_away_bet = np.maximum(adjusted_away_bet, 0.1)

    print("\nPer-competition bias correction (A residuals -> B)")
    print(bias.round(3).to_string())
    print(f"\nB raw:   home={predicted_home_bet[~a_mask].mean():.3f} away={predicted_away_bet[~a_mask].mean():.3f}")
    print(f"B calib: home={adjusted_home_bet[~a_mask].mean():.3f} away={adjusted_away_bet[~a_mask].mean():.3f}")
    mae_h_raw = np.abs(betting_match_features.loc[~a_mask, "home_corners"].values - predicted_home_bet[~a_mask]).mean()
    mae_h_cal = np.abs(betting_match_features.loc[~a_mask, "home_corners"].values - adjusted_home_bet[~a_mask]).mean()
    mae_a_raw = np.abs(betting_match_features.loc[~a_mask, "away_corners"].values - predicted_away_bet[~a_mask]).mean()
    mae_a_cal = np.abs(betting_match_features.loc[~a_mask, "away_corners"].values - adjusted_away_bet[~a_mask]).mean()
    print(f"B MAE home: raw={mae_h_raw:.4f} calib={mae_h_cal:.4f} delta={mae_h_cal - mae_h_raw:+.4f}")
    print(f"B MAE away: raw={mae_a_raw:.4f} calib={mae_a_cal:.4f} delta={mae_a_cal - mae_a_raw:+.4f}")

    market_home_prob_all = betting_match_features["p_h_1x2"].fillna(MARKET_TWO_WAY_FILL).values
    market_away_prob_all = betting_match_features["p_a_1x2"].fillna(MARKET_TWO_WAY_FILL).values
    rolling_home_std_all = betting_match_features["home_corners_for_std_rolling_window_20"].fillna(global_std_c).values
    rolling_away_std_all = betting_match_features["away_corners_for_std_rolling_window_20"].fillna(global_std_c).values
    predicted_home_variance = dispersion_home.predict(adjusted_home_bet, np.abs(market_home_prob_all - 0.5), rolling_home_std_all)
    predicted_away_variance = dispersion_away.predict(adjusted_away_bet, np.abs(market_away_prob_all - 0.5), rolling_away_std_all)

    print("\nDispersion calibration audit on partition A (home)")
    predicted_home_variance_a = dispersion_home.predict(
        predicted_home_partition_a,
        market_certainty_home_a,
        rolling_home_std_a,
    )
    audit = pd.DataFrame(
        {
            "pred_bucket": pd.qcut(predicted_home_partition_a, 5, duplicates="drop"),
            "realised_sq_resid": (actual_home_partition_a - predicted_home_partition_a) ** 2,
            "pred_variance": predicted_home_variance_a,
        }
    )
    audit_grouped = audit.groupby("pred_bucket", observed=True).agg(
        n=("realised_sq_resid", "size"),
        realised_var=("realised_sq_resid", "mean"),
        pred_var=("pred_variance", "mean"),
    )
    audit_grouped["ratio"] = audit_grouped["realised_var"] / audit_grouped["pred_var"]
    print(audit_grouped.round(3).to_string())

    print("\nEdge preservation: (pred_home + pred_away) vs OU line on partition B")
    holdout = betting_match_features[~a_mask].copy()
    predicted_total_b = (adjusted_home_bet + adjusted_away_bet)[~a_mask]
    diff_total = (predicted_total_b - holdout["ou_line"]).dropna()
    print(f"B with OU line: n={len(diff_total)} median |pred_total - OU_line|={float(diff_total.abs().median()):.3f}")

    print("\nSpot-check match 12449945")
    row = betting_match_features[betting_match_features["match_id"] == 12449945]
    if len(row):
        pos = list(betting_match_features["match_id"]).index(12449945)
        print(
            f"pred_home={adjusted_home_bet[pos]:.2f} pred_away={adjusted_away_bet[pos]:.2f} "
            f"pred_diff={adjusted_home_bet[pos] - adjusted_away_bet[pos]:+.2f}"
        )
        print(f"actual: home={int(row.iloc[0].home_corners)} away={int(row.iloc[0].away_corners)}")
        print(f"sigma2_home={predicted_home_variance[pos]:.2f} sigma2_away={predicted_away_variance[pos]:.2f}")

    calibration = {
        "dispersion_home": dispersion_home,
        "dispersion_away": dispersion_away,
        "a_mask": a_mask,
        "predicted_home_bet_adjusted": adjusted_home_bet,
        "predicted_away_bet_adjusted": adjusted_away_bet,
        "predicted_home_variance": predicted_home_variance,
        "predicted_away_variance": predicted_away_variance,
        "global_std_c": global_std_c,
        "market_home_prob_all": market_home_prob_all,
        "market_away_prob_all": market_away_prob_all,
        "rolling_home_std_all": rolling_home_std_all,
        "rolling_away_std_all": rolling_away_std_all,
        "actual_home_partition_a": actual_home_partition_a,
        "actual_away_partition_a": actual_away_partition_a,
        "predicted_home_partition_a": predicted_home_partition_a,
        "predicted_away_partition_a": predicted_away_partition_a,
    }
    return betting_match_features, adjusted_home_bet, adjusted_away_bet, predicted_home_variance, predicted_away_variance, calibration


def make_mean_calibration_plot(
    output_dir: Path,
    model_data: pd.DataFrame,
    betting_match_features: pd.DataFrame,
    y_val_h: pd.Series,
    y_val_a: pd.Series,
    predicted_home_val: np.ndarray,
    predicted_away_val: np.ndarray,
) -> None:
    """Draw the main mean-calibration plots for Q1."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, y_series, mu, name in [
        (axes[0], y_val_h, predicted_home_val, "Home"),
        (axes[1], y_val_a, predicted_away_val, "Away"),
    ]:
        y = np.asarray(y_series, float)
        cal = pd.DataFrame({"mu": mu, "y": y})
        cal["decile"] = pd.qcut(cal["mu"], 10, duplicates="drop", labels=False)
        grouped = cal.groupby("decile").agg(mean_mu=("mu", "mean"), mean_y=("y", "mean"), n=("y", "size"))

        lo = min(grouped["mean_mu"].min(), grouped["mean_y"].min()) * 0.95
        hi = max(grouped["mean_mu"].max(), grouped["mean_y"].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect")
        ax.scatter(grouped["mean_mu"], grouped["mean_y"], s=grouped["n"] * 0.4, alpha=0.85, zorder=3)
        ax.set_xlabel("Mean predicted mu")
        ax.set_ylabel("Mean actual corners")
        ax.set_title(f"{name} corners - mean calibration (val set)")
        ax.legend(fontsize=8)

        bias = float((y - mu).mean())
        n = len(y)
        early_bias = float((y[: n // 3] - mu[: n // 3]).mean())
        late_bias = float((y[2 * n // 3 :] - mu[2 * n // 3 :]).mean())
        print(
            f"{name:6s}: global bias={bias:+.3f} early_val={early_bias:+.3f} "
            f"late_val={late_bias:+.3f} (pred_std={mu.std():.2f} actual_std={y.std():.2f})"
        )

    plt.tight_layout()
    plt.savefig(output_dir / "q1_mean_calibration.png", dpi=150)
    plt.close(fig)

    all_trend = pd.concat(
        [
            model_data[["date_time", "home_corners", "away_corners"]],
            betting_match_features[["date_time", "home_corners", "away_corners"]],
        ],
        ignore_index=True,
    ).dropna()
    all_trend["ym"] = all_trend["date_time"].dt.to_period("M")
    monthly = all_trend.groupby("ym").agg(
        mean_home=("home_corners", "mean"),
        mean_away=("away_corners", "mean"),
        n=("home_corners", "size"),
    ).reset_index()
    monthly["date"] = monthly["ym"].dt.to_timestamp()

    fig2, ax2 = plt.subplots(figsize=(11, 3))
    ax2.plot(monthly["date"], monthly["mean_home"], label="Home mean", linewidth=1.5)
    ax2.plot(monthly["date"], monthly["mean_away"], label="Away mean", linewidth=1.5)
    ax2.axvline(
        betting_match_features["date_time"].min(),
        color="red",
        linestyle="--",
        linewidth=1,
        label="Betting period start",
    )
    ax2.set_title("Monthly mean corners - training + betting periods")
    ax2.set_ylabel("Mean corners")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig2.autofmt_xdate()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q1_corner_trend.png", dpi=150)
    plt.close(fig2)
    print("Saved q1_mean_calibration.png and q1_corner_trend.png")


def make_distribution_plot(
    output_dir: Path,
    val_feat: pd.DataFrame,
    y_val_h: pd.Series,
    y_val_a: pd.Series,
    predicted_home_val: np.ndarray,
    predicted_away_val: np.ndarray,
    calibration: dict[str, object],
) -> dict[str, float]:
    """Draw the validation distribution check plots."""
    dispersion_home = calibration["dispersion_home"]
    dispersion_away = calibration["dispersion_away"]
    global_std_c = float(calibration["global_std_c"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    inflation_factors: dict[str, float] = {}

    for ax, y_series, mu, std_col, disp_model, name, key in [
        (
            axes[0],
            y_val_h,
            predicted_home_val,
            "home_corners_for_std_rolling_window_20",
            dispersion_home,
            "Home",
            "h",
        ),
        (
            axes[1],
            y_val_a,
            predicted_away_val,
            "away_corners_for_std_rolling_window_20",
            dispersion_away,
            "Away",
            "a",
        ),
    ]:
        y = np.asarray(y_series, float)
        y_int = y.astype(int)
        k_max = min(int(y.max()) + 2, 30)
        ks = np.arange(k_max)

        counts = np.bincount(y_int, minlength=k_max)[:k_max].astype(float)
        pmf_emp = counts / counts.sum()
        lam = float(y.mean())
        var_y = float(np.var(y))
        vm_emp = var_y / lam
        pmf_pois = poisson.pmf(ks, lam)

        if var_y > lam:
            n_emp = lam**2 / (var_y - lam)
            p_emp = lam / var_y
            pmf_nb_emp = nbinom.pmf(ks, n_emp, p_emp)
        else:
            pmf_nb_emp = pmf_pois.copy()

        val_std = val_feat[std_col].fillna(global_std_c).values
        sigma2_model = disp_model.predict(mu, np.zeros(len(mu)), val_std)
        var_model = float(sigma2_model.mean())
        vm_model = var_model / lam
        if var_model > lam:
            n_model = lam**2 / (var_model - lam)
            p_model = lam / var_model
            pmf_nb_model = nbinom.pmf(ks, n_model, p_model)
        else:
            pmf_nb_model = pmf_pois.copy()

        var_residual = float(np.mean((y - mu) ** 2))
        vm_residual = var_residual / lam
        k_inflation = var_residual / var_model
        inflation_factors[key] = k_inflation
        if var_residual > lam:
            n_res = lam**2 / (var_residual - lam)
            p_res = lam / var_residual
            pmf_nb_residual = nbinom.pmf(ks, n_res, p_res)
        else:
            pmf_nb_residual = pmf_pois.copy()

        ax.bar(ks - 0.12, pmf_emp, width=0.25, alpha=0.65, label="Empirical", color="steelblue")
        ax.step(ks, pmf_nb_emp, where="mid", color="green", lw=2.2, linestyle="-", label=f"NB fit V/M={vm_emp:.2f}")
        ax.step(ks, pmf_nb_model, where="mid", color="orange", lw=2, linestyle="--", label=f"NB model V/M={vm_model:.2f}")
        ax.step(
            ks,
            pmf_nb_residual,
            where="mid",
            color="red",
            lw=2,
            linestyle="-.",
            label=f"NB residual V/M={vm_residual:.2f} k={k_inflation:.2f}",
        )
        ax.step(ks, pmf_pois, where="mid", color="grey", lw=1.2, linestyle=":", label="Poisson")
        ax.set_xlim(0, min(k_max - 1, 22))
        ax.set_title(f"{name} corners - distribution check (val set)")
        ax.set_xlabel("Corners")
        ax.set_ylabel("Probability")
        ax.legend(fontsize=7.5)

        print(
            f"{name}: empirical V/M={vm_emp:.2f} model V/M={vm_model:.2f} "
            f"residual V/M={vm_residual:.2f} -> inflation k={k_inflation:.3f}"
        )
        print(
            f"       empirical std={np.sqrt(var_y):.2f} "
            f"model sigma={np.sqrt(var_model):.2f} residual sigma={np.sqrt(var_residual):.2f}"
        )

    plt.tight_layout()
    plt.savefig(output_dir / "q1_distribution_check.png", dpi=150)
    plt.close(fig)
    print("Saved q1_distribution_check.png")
    return inflation_factors


def apply_global_mean_correction(
    model_data: pd.DataFrame,
    betting_match_features: pd.DataFrame,
    calibration: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the late-window mean correction to betting predictions."""
    a_mask = calibration["a_mask"]
    adjusted_home_bet = np.asarray(calibration["predicted_home_bet_adjusted"]).copy()
    adjusted_away_bet = np.asarray(calibration["predicted_away_bet_adjusted"]).copy()
    dispersion_home = calibration["dispersion_home"]
    dispersion_away = calibration["dispersion_away"]
    market_home_prob_all = np.asarray(calibration["market_home_prob_all"])
    market_away_prob_all = np.asarray(calibration["market_away_prob_all"])
    rolling_home_std_all = np.asarray(calibration["rolling_home_std_all"])
    rolling_away_std_all = np.asarray(calibration["rolling_away_std_all"])

    last_train_date = model_data["date_time"].max()
    six_months_ago = last_train_date - pd.DateOffset(months=6)
    late_train = model_data[model_data["date_time"] > six_months_ago]
    global_corr_h = float(late_train["home_corners"].mean() - model_data["home_corners"].mean())
    global_corr_a = float(late_train["away_corners"].mean() - model_data["away_corners"].mean())

    print_heading("Late-Window Mean Correction")
    print(
        "Global mean correction (late 6-month training vs overall training)\n"
        f"Period: {late_train.date_time.min().date()} - {late_train.date_time.max().date()} n={len(late_train)}\n"
        f"Home: overall={model_data.home_corners.mean():.3f} "
        f"late={late_train.home_corners.mean():.3f} correction={global_corr_h:+.3f}\n"
        f"Away: overall={model_data.away_corners.mean():.3f} "
        f"late={late_train.away_corners.mean():.3f} correction={global_corr_a:+.3f}"
    )

    adjusted_home_bet[a_mask] = np.maximum(adjusted_home_bet[a_mask] + global_corr_h, 0.1)
    adjusted_away_bet[a_mask] = np.maximum(adjusted_away_bet[a_mask] + global_corr_a, 0.1)
    predicted_home_variance = dispersion_home.predict(adjusted_home_bet, np.abs(market_home_prob_all - 0.5), rolling_home_std_all)
    predicted_away_variance = dispersion_away.predict(adjusted_away_bet, np.abs(market_away_prob_all - 0.5), rolling_away_std_all)

    print("\nBetting set: predicted vs actual means after calibration")
    for label, mask in [("A", a_mask), ("B", ~a_mask)]:
        pred_home = adjusted_home_bet[mask].mean()
        pred_away = adjusted_away_bet[mask].mean()
        actual_home = betting_match_features.loc[mask, "home_corners"].mean()
        actual_away = betting_match_features.loc[mask, "away_corners"].mean()
        print(
            f"Partition {label}: pred home={pred_home:.3f} (actual {actual_home:.3f}, bias {pred_home - actual_home:+.3f}) "
            f"pred away={pred_away:.3f} (actual {actual_away:.3f}, bias {pred_away - actual_away:+.3f})"
        )
    print(f"sigma2 home: median={np.median(predicted_home_variance):.2f} away: median={np.median(predicted_away_variance):.2f}")
    return adjusted_home_bet, adjusted_away_bet, predicted_home_variance, predicted_away_variance


def apply_market_teacher_adjustment(
    betting_match_features: pd.DataFrame,
    predicted_home_bet: np.ndarray,
    predicted_away_bet: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Blend the base Q1 means with the market-teacher means on partition B."""
    market_frame = betting_match_features.copy()
    market_frame["base_pred_home"] = predicted_home_bet
    market_frame["base_pred_away"] = predicted_away_bet
    market_frame["base_pred_total"] = predicted_home_bet + predicted_away_bet
    market_frame["base_pred_diff"] = predicted_home_bet - predicted_away_bet

    inferred = market_frame.apply(infer_market_means_for_row, axis=1)
    market_frame["market_target_home"] = [item[0] if item is not None else np.nan for item in inferred]
    market_frame["market_target_away"] = [item[1] if item is not None else np.nan for item in inferred]
    market_frame["has_market_target"] = market_frame["market_target_home"].notna() & market_frame["market_target_away"].notna()

    a_mask = market_frame["partition"] == "A"
    b_mask = market_frame["partition"] == "B"
    usable_a = market_frame[a_mask & market_frame["has_market_target"]].sort_values("date_time").reset_index()

    print_heading("Market Teacher")
    print(
        f"usable 1X2+OU inferred targets: total={int(market_frame['has_market_target'].sum())} "
        f"A={int((a_mask & market_frame['has_market_target']).sum())} "
        f"B={int((b_mask & market_frame['has_market_target']).sum())}"
    )

    if len(usable_a) < 120:
        print("Not enough usable A-market targets to fit a stable teacher; skipping market adjustment.")
        market_frame["market_teacher_home"] = np.nan
        market_frame["market_teacher_away"] = np.nan
        return market_frame, predicted_home_bet.copy(), predicted_away_bet.copy()

    teacher_cols_home = FEATURE_COLS + ["base_pred_home", "base_pred_away", "base_pred_total", "base_pred_diff"]
    teacher_cols_away = list(FEATURE_COLS)
    x_a_home = usable_a[teacher_cols_home]
    x_a_away = usable_a[teacher_cols_away]
    y_a_home = usable_a["market_target_home"]
    y_a_away = usable_a["market_target_away"]

    n_splits = min(4, max(2, len(usable_a) // 120))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_home = np.full(len(usable_a), np.nan)
    oof_away = np.full(len(usable_a), np.nan)

    for train_idx, val_idx in tscv.split(x_a_home):
        x_train_home, x_val_home = x_a_home.iloc[train_idx], x_a_home.iloc[val_idx]
        x_train_away, x_val_away = x_a_away.iloc[train_idx], x_a_away.iloc[val_idx]
        y_train_home, y_val_home = y_a_home.iloc[train_idx], y_a_home.iloc[val_idx]
        y_train_away, y_val_away = y_a_away.iloc[train_idx], y_a_away.iloc[val_idx]

        model_home = fit_market_teacher(x_train_home, y_train_home, x_val_home, y_val_home)
        model_away = fit_market_teacher(x_train_away, y_train_away, x_val_away, y_val_away)
        oof_home[val_idx] = np.maximum(model_home.predict(x_val_home), 0.1)
        oof_away[val_idx] = np.maximum(model_away.predict(x_val_away), 0.1)

    valid_oof = np.isfinite(oof_home) & np.isfinite(oof_away)
    weight_home = 0.0
    weight_away = 0.0
    if valid_oof.sum() >= max(80, len(usable_a) // 3):
        base_home_a = usable_a.loc[valid_oof, "base_pred_home"].to_numpy()
        base_away_a = usable_a.loc[valid_oof, "base_pred_away"].to_numpy()
        actual_home_a = usable_a.loc[valid_oof, "home_corners"].to_numpy()
        actual_away_a = usable_a.loc[valid_oof, "away_corners"].to_numpy()
        weight_home = optimise_blend_weight(base_home_a, oof_home[valid_oof], actual_home_a)
        weight_away = optimise_blend_weight(base_away_a, oof_away[valid_oof], actual_away_a)

        print(
            f"A OOF weight_home={weight_home:.2f} weight_away={weight_away:.2f}\n"
            f"A OOF base MAE home={mean_absolute_error(actual_home_a, base_home_a):.4f} "
            f"teacher={mean_absolute_error(actual_home_a, oof_home[valid_oof]):.4f} "
            f"blend={mean_absolute_error(actual_home_a, (1 - weight_home) * base_home_a + weight_home * oof_home[valid_oof]):.4f}\n"
            f"A OOF base MAE away={mean_absolute_error(actual_away_a, base_away_a):.4f} "
            f"teacher={mean_absolute_error(actual_away_a, oof_away[valid_oof]):.4f} "
            f"blend={mean_absolute_error(actual_away_a, (1 - weight_away) * base_away_a + weight_away * oof_away[valid_oof]):.4f}"
        )
    else:
        print("Not enough chronological OOF coverage inside A; fitted teacher will be diagnostic only.")

    split_point = max(1, int(len(usable_a) * 0.8))
    x_train_home_final = x_a_home.iloc[:split_point]
    x_val_home_final = x_a_home.iloc[split_point:] if split_point < len(usable_a) else None
    x_train_away_final = x_a_away.iloc[:split_point]
    x_val_away_final = x_a_away.iloc[split_point:] if split_point < len(usable_a) else None
    y_train_home_final = y_a_home.iloc[:split_point]
    y_val_home_final = y_a_home.iloc[split_point:] if split_point < len(usable_a) else None
    y_train_away_final = y_a_away.iloc[:split_point]
    y_val_away_final = y_a_away.iloc[split_point:] if split_point < len(usable_a) else None

    final_home_model = fit_market_teacher(
        x_train_home_final,
        y_train_home_final,
        x_val_home_final,
        y_val_home_final,
    )
    final_away_model = fit_market_teacher(
        x_train_away_final,
        y_train_away_final,
        x_val_away_final,
        y_val_away_final,
    )

    teacher_all_home = np.maximum(final_home_model.predict(market_frame[teacher_cols_home]), 0.1)
    teacher_all_away = np.maximum(final_away_model.predict(market_frame[teacher_cols_away]), 0.1)
    market_frame["market_teacher_home"] = teacher_all_home
    market_frame["market_teacher_away"] = teacher_all_away

    adjusted_home = predicted_home_bet.copy()
    adjusted_away = predicted_away_bet.copy()
    adjusted_home[b_mask.values] = (
        (1 - weight_home) * adjusted_home[b_mask.values] + weight_home * teacher_all_home[b_mask.values]
    )
    adjusted_away[b_mask.values] = (
        (1 - weight_away) * adjusted_away[b_mask.values] + weight_away * teacher_all_away[b_mask.values]
    )

    if b_mask.sum():
        actual_home_b = market_frame.loc[b_mask, "home_corners"].to_numpy()
        actual_away_b = market_frame.loc[b_mask, "away_corners"].to_numpy()
        print(
            f"B diagnostic MAE home: base={mean_absolute_error(actual_home_b, predicted_home_bet[b_mask.values]):.4f} "
            f"blend={mean_absolute_error(actual_home_b, adjusted_home[b_mask.values]):.4f}\n"
            f"B diagnostic MAE away: base={mean_absolute_error(actual_away_b, predicted_away_bet[b_mask.values]):.4f} "
            f"blend={mean_absolute_error(actual_away_b, adjusted_away[b_mask.values]):.4f}"
        )
    print("Market teacher fitted on partition A only and applied to partition B only.")
    print("Teacher inputs: home keeps base Q1 preds; away uses non-market features only.")
    return market_frame, adjusted_home, adjusted_away


def save_outputs(
    output_dir: Path,
    val_feat: pd.DataFrame,
    betting_match_features: pd.DataFrame,
    results: pd.DataFrame,
    predicted_home_val: np.ndarray,
    predicted_away_val: np.ndarray,
    predicted_home_bet: np.ndarray,
    predicted_away_bet: np.ndarray,
    predicted_home_variance: np.ndarray,
    predicted_away_variance: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the Q1 validation and betting prediction files."""
    model_path = write_csv_with_fallback(results, output_dir / "q1_model_comparison.csv")

    val_keep = ["match_id", "date_time", "competition_id", "season_id", "home_corners", "away_corners"]
    val_pred = val_feat[val_keep].copy()
    val_pred["pred_home"] = predicted_home_val
    val_pred["pred_away"] = predicted_away_val
    val_path = write_csv_with_fallback(val_pred, output_dir / "q1_validation_predictions.csv")

    export_cols = [
        "match_id",
        "date_time",
        "competition_id",
        "season_id",
        "home_corners",
        "away_corners",
        "partition",
        "pred_home_q10",
        "pred_home_q50",
        "pred_home_q90",
        "pred_away_q10",
        "pred_away_q50",
        "pred_away_q90",
        "quantile_sigma2_home",
        "quantile_sigma2_away",
    ]
    bet_pred = betting_match_features[
        [col for col in export_cols if col in betting_match_features.columns]
    ].copy()
    bet_pred["pred_home_corners"] = predicted_home_bet
    bet_pred["pred_away_corners"] = predicted_away_bet
    bet_pred["pred_corner_diff"] = predicted_home_bet - predicted_away_bet
    bet_pred["sigma2_home"] = predicted_home_variance
    bet_pred["sigma2_away"] = predicted_away_variance
    preferred_column_order = [
        "match_id",
        "date_time",
        "competition_id",
        "season_id",
        "home_corners",
        "away_corners",
        "partition",
        "pred_home_corners",
        "pred_away_corners",
        "pred_corner_diff",
        "sigma2_home",
        "sigma2_away",
        "pred_home_q10",
        "pred_home_q50",
        "pred_home_q90",
        "pred_away_q10",
        "pred_away_q50",
        "pred_away_q90",
        "quantile_sigma2_home",
        "quantile_sigma2_away",
    ]
    bet_pred = bet_pred[[col for col in preferred_column_order if col in bet_pred.columns]]
    bet_path = write_csv_with_fallback(bet_pred, output_dir / "q1_betting_match_predictions.csv")
    outputs_q1_dir = output_dir / "outputs" / "q1"
    outputs_q1_dir.mkdir(parents=True, exist_ok=True)
    write_csv_with_fallback(bet_pred, outputs_q1_dir / "q1_betting_match_predictions.csv")

    print_heading("Save Outputs")
    print(f"Saved outputs: {model_path.name}, {val_path.name}, {bet_path.name}")
    print(
        f"Betting partition counts: A={(betting_match_features.partition == 'A').sum()} "
        f"B={(betting_match_features.partition == 'B').sum()}\n"
        f"pred_corner_diff: std={(predicted_home_bet - predicted_away_bet).std():.3f} "
        f"range=[{(predicted_home_bet - predicted_away_bet).min():.2f},{(predicted_home_bet - predicted_away_bet).max():.2f}]\n"
        f"sigma2_home: median={np.median(predicted_home_variance):.2f} sigma2_away: median={np.median(predicted_away_variance):.2f}"
    )
    return val_pred, bet_pred


def run_pipeline(data_dir: str | Path = ".", save_outputs_flag: bool = True, make_plots: bool = True) -> PipelineArtifacts:
    """Run the full Q1 pipeline."""
    data_path = Path(data_dir)
    train, betting, all_matches, model_data, betting_matches = load_inputs(data_path)
    _, bound = baseline_diagnostics(model_data)
    strength_features = build_strength_features(all_matches, model_data, betting_matches)
    team_games = build_team_games(model_data)
    team_games = add_rolling_features(team_games)
    features, betting_match_features = build_match_features(
        model_data,
        betting,
        betting_matches,
        team_games,
        strength_features,
    )
    features, betting_match_features, global_std_c = fill_missing_features(
        model_data,
        features,
        betting_match_features,
    )
    train_feat, val_feat, betting_match_features = split_train_val(features, betting_match_features)
    home_model, away_model, predicted_home_val, predicted_away_val, _, _, results = fit_mean_models(train_feat, val_feat)
    gap = distance_to_floor(bound, results)

    actual_home_val = val_feat["home_corners"]
    actual_away_val = val_feat["away_corners"]
    betting_match_features, predicted_home_bet, predicted_away_bet, predicted_home_variance, predicted_away_variance, calibration = stage3_calibration(
        betting,
        betting_match_features,
        val_feat,
        model_data,
        home_model,
        away_model,
        predicted_home_val,
        predicted_away_val,
        actual_home_val,
        actual_away_val,
        global_std_c,
    )

    if make_plots:
        print_heading("Calibration Plots")
        make_mean_calibration_plot(
            data_path,
            model_data,
            betting_match_features,
            actual_home_val,
            actual_away_val,
            predicted_home_val,
            predicted_away_val,
        )
        make_distribution_plot(
            data_path,
            val_feat,
            actual_home_val,
            actual_away_val,
            predicted_home_val,
            predicted_away_val,
            calibration,
        )

    predicted_home_bet, predicted_away_bet, predicted_home_variance, predicted_away_variance = apply_global_mean_correction(
        model_data,
        betting_match_features,
        calibration,
    )
    betting_match_features, predicted_home_bet, predicted_away_bet = apply_market_teacher_adjustment(
        betting_match_features,
        predicted_home_bet,
        predicted_away_bet,
    )
    betting_match_features = attach_market_quantile_features(
        betting_match_features,
        predicted_home_bet,
        predicted_away_bet,
        global_std_c,
    )
    predicted_home_variance = calibration["dispersion_home"].predict(
        predicted_home_bet,
        np.abs(np.asarray(calibration["market_home_prob_all"]) - 0.5),
        np.asarray(calibration["rolling_home_std_all"]),
    )
    predicted_away_variance = calibration["dispersion_away"].predict(
        predicted_away_bet,
        np.abs(np.asarray(calibration["market_away_prob_all"]) - 0.5),
        np.asarray(calibration["rolling_away_std_all"]),
    )

    if save_outputs_flag:
        val_predictions, bet_predictions = save_outputs(
            data_path,
            val_feat,
            betting_match_features,
            results,
            predicted_home_val,
            predicted_away_val,
            predicted_home_bet,
            predicted_away_bet,
            predicted_home_variance,
            predicted_away_variance,
        )
    else:
        val_predictions = pd.DataFrame()
        bet_predictions = pd.DataFrame()

    return PipelineArtifacts(
        model_data=model_data,
        betting_matches=betting_matches,
        features=features,
        betting_match_features=betting_match_features,
        train_feat=train_feat,
        val_feat=val_feat,
        results=results,
        gap=gap,
        val_predictions=val_predictions,
        bet_predictions=bet_predictions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Q1 football corners pipeline.")
    parser.add_argument("--data-dir", default=".", help="Directory containing parquet and csv outputs.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing CSV outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing PNG diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        save_outputs_flag=not args.no_save,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
