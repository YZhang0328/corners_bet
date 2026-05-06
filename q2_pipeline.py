from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, norm
from sklearn.isotonic import IsotonicRegression


matplotlib.use("Agg")

STAKE = 1.0
DEFAULT_EV_THRESHOLD = 0.03
DEFAULT_MARKET_THRESHOLDS = {"1X2": 0.040, "HC": 0.023, "OU": 0.030}
DEFAULT_MONTE_CARLO_RUNS = 10_000
CALIBRATION_MIN_SAMPLES = 120
CALIBRATION_SHRINKAGE = 500.0
TAIL_SHRINK_MARKETS = ("1X2", "HC")
TAIL_SHRINK_MAX_LAMBDA = 6.0
TAIL_SCALE_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
NORMAL_Q10_Q90_SPAN = 2.5631031310892007
DEFAULT_INITIAL_BANKROLL = 100.0
MAX_FULL_KELLY_FRACTION = 0.10
DEFAULT_STAKING_PLAN = {
    "baseline_fixed": {"mode": "fixed", "fraction": 1.0},
    "kelly_100": {"mode": "kelly", "fraction": 1.0},
    "kelly_50": {"mode": "kelly", "fraction": 0.5},
    "kelly_25": {"mode": "kelly", "fraction": 0.25},
}


@dataclass
class Q2Artifacts:
    """Container for the main Q2 intermediate and final tables."""

    partition_a_bets: pd.DataFrame
    partition_b_bets: pd.DataFrame
    selected_bets: pd.DataFrame
    base_calibration_report: pd.DataFrame
    partition_summary: pd.DataFrame
    tail_scale_report: pd.DataFrame
    run_summary: pd.DataFrame
    staking_summary: pd.DataFrame


def print_heading(title: str) -> None:
    """Print a section header for Q2 console output."""
    print(f"\n=== {title} ===")


def read_prediction_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Q1 betting predictions, market prices, and observed results."""
    latest_prediction_path = data_dir / "q1_betting_match_predictions.latest.csv"
    candidate_paths = [
        latest_prediction_path,
        data_dir / "q1_betting_match_predictions.csv",
        data_dir / "outputs" / "q1" / "q1_betting_match_predictions.csv",
    ]
    prediction_path = next((path for path in candidate_paths if path.exists()), data_dir / "q1_betting_match_predictions.csv")
    predicted_matches = pd.read_csv(prediction_path, parse_dates=["date_time"])
    market_prices = pd.read_csv(data_dir / "corners_prices.csv", parse_dates=["date_time"])
    observed_results = pd.read_csv(data_dir / "corners_prices_results.csv")
    print_heading("Load Q2 Inputs")
    print(
        f"predictions={prediction_path.name} rows={len(predicted_matches)} | "
        f"market rows={len(market_prices)} | result rows={len(observed_results)}"
    )
    return predicted_matches, market_prices, observed_results


def attach_quantile_aware_distribution_inputs(predicted_matches: pd.DataFrame) -> pd.DataFrame:
    """Attach quantile diagnostics while keeping raw Q1 mean/variance as the primary moments."""
    prepared = predicted_matches.copy()
    prepared["q2_home_center"] = prepared["pred_home_corners"]
    prepared["q2_away_center"] = prepared["pred_away_corners"]
    prepared["q2_home_variance"] = prepared["sigma2_home"]
    prepared["q2_away_variance"] = prepared["sigma2_away"]
    prepared["quantile_home_center"] = prepared.get("pred_home_q50", prepared["pred_home_corners"])
    prepared["quantile_away_center"] = prepared.get("pred_away_q50", prepared["pred_away_corners"])
    prepared["quantile_home_variance"] = prepared.get("quantile_sigma2_home", prepared["sigma2_home"])
    prepared["quantile_away_variance"] = prepared.get("quantile_sigma2_away", prepared["sigma2_away"])

    print_heading("Quantile-Aware Moments")
    print(
        f"q50 coverage home={prepared['quantile_home_center'].notna().mean():.1%} "
        f"away={prepared['quantile_away_center'].notna().mean():.1%} | "
        f"median quantile/raw variance ratio home={np.median(prepared['quantile_home_variance'] / prepared['sigma2_home']):.3f} "
        f"away={np.median(prepared['quantile_away_variance'] / prepared['sigma2_away']):.3f}"
    )
    return prepared


def quantile_tail_distance(prediction_row: pd.Series, market_type: str, market_line: float, side_name: str) -> float:
    """Measure how far a candidate side sits outside Q1's empirical q10-q90 envelope."""
    required_cols = {
        "pred_home_q10",
        "pred_home_q50",
        "pred_home_q90",
        "pred_away_q10",
        "pred_away_q50",
        "pred_away_q90",
    }
    if not required_cols.issubset(prediction_row.index):
        return 0.0
    if prediction_row[list(required_cols)].isna().any():
        return 0.0

    total_q10 = float(prediction_row["pred_home_q10"] + prediction_row["pred_away_q10"])
    total_q90 = float(prediction_row["pred_home_q90"] + prediction_row["pred_away_q90"])
    total_width = max(total_q90 - total_q10, 1e-6)

    diff_q10 = float(prediction_row["pred_home_q10"] - prediction_row["pred_away_q90"])
    diff_q50 = float(prediction_row["pred_home_q50"] - prediction_row["pred_away_q50"])
    diff_q90 = float(prediction_row["pred_home_q90"] - prediction_row["pred_away_q10"])
    diff_width = max(diff_q90 - diff_q10, 1e-6)

    if market_type == "OU":
        if side_name == "over":
            return max(market_line - total_q90, 0.0) / total_width
        if side_name == "under":
            return max(total_q10 - market_line, 0.0) / total_width
        return 0.0

    if market_type == "HC":
        if side_name == "home":
            return max(market_line - diff_q90, 0.0) / diff_width
        if side_name == "away":
            return max(diff_q10 - market_line, 0.0) / diff_width
        return 0.0

    if side_name == "home":
        return max(0.0 - diff_q90, 0.0) / diff_width
    if side_name == "away":
        return max(diff_q10 - 0.0, 0.0) / diff_width
    half_width = max(diff_width / 2.0, 1e-6)
    return max(abs(diff_q50) - half_width, 0.0) / half_width


def estimate_shared_correlation(calibration_matches: pd.DataFrame) -> dict[str, float]:
    """Estimate one shared home-away corner correlation from partition A."""
    clean_rows = calibration_matches.dropna(
        subset=[
            "q2_home_center",
            "q2_away_center",
            "q2_home_variance",
            "q2_away_variance",
            "home_corners",
            "away_corners",
        ]
    ).copy()
    if clean_rows.empty:
        return {"rho": -0.20, "k_total": np.nan, "k_diff": np.nan, "n": 0}

    independent_variance = clean_rows["q2_home_variance"] + clean_rows["q2_away_variance"]
    covariance_term = np.sqrt(np.maximum(clean_rows["q2_home_variance"], 1e-6) * np.maximum(clean_rows["q2_away_variance"], 1e-6))
    actual_total = clean_rows["home_corners"] + clean_rows["away_corners"]
    predicted_total = clean_rows["q2_home_center"] + clean_rows["q2_away_center"]
    actual_diff = clean_rows["home_corners"] - clean_rows["away_corners"]
    predicted_diff = clean_rows["q2_home_center"] - clean_rows["q2_away_center"]

    total_residual_variance = float(np.mean((actual_total - predicted_total) ** 2))
    diff_residual_variance = float(np.mean((actual_diff - predicted_diff) ** 2))
    average_independent_variance = float(independent_variance.mean())
    average_covariance_term = float(np.maximum(covariance_term.mean(), 1e-6))

    rho_total = (total_residual_variance - average_independent_variance) / (2.0 * average_covariance_term)
    rho_diff = (average_independent_variance - diff_residual_variance) / (2.0 * average_covariance_term)
    shared_rho = float(np.clip((rho_total + rho_diff) / 2.0, -0.95, 0.95))
    return {
        "rho": shared_rho,
        "k_total": total_residual_variance / np.maximum(average_independent_variance, 1e-6),
        "k_diff": diff_residual_variance / np.maximum(average_independent_variance, 1e-6),
        "n": len(clean_rows),
    }


def negative_binomial_params(predicted_mean: float, predicted_variance: float) -> tuple[float, float]:
    """Convert count moments into scipy's negative-binomial parameterization."""
    stable_mean = float(np.maximum(predicted_mean, 1e-6))
    stable_variance = float(np.maximum(predicted_variance, stable_mean + 1e-6))
    success_prob = stable_mean / stable_variance
    num_failures = stable_mean * success_prob / (1 - success_prob)
    return num_failures, success_prob


def combined_total_moments(
    predicted_home_corners: float,
    predicted_home_variance: float,
    predicted_away_corners: float,
    predicted_away_variance: float,
    shared_rho: float,
) -> tuple[float, float]:
    """Combine home and away predictions into total-corners moments."""
    predicted_total = float(np.maximum(predicted_home_corners + predicted_away_corners, 1e-6))
    covariance_adjustment = 2.0 * shared_rho * np.sqrt(
        np.maximum(predicted_home_variance, 1e-6) * np.maximum(predicted_away_variance, 1e-6)
    )
    predicted_total_variance = float(
        np.maximum(predicted_total + 1e-6, predicted_home_variance + predicted_away_variance + covariance_adjustment)
    )
    return predicted_total, predicted_total_variance


def combined_diff_moments(
    predicted_home_corners: float,
    predicted_home_variance: float,
    predicted_away_corners: float,
    predicted_away_variance: float,
    shared_rho: float,
) -> tuple[float, float]:
    """Combine home and away predictions into corner-difference moments."""
    predicted_diff = float(predicted_home_corners - predicted_away_corners)
    covariance_adjustment = 2.0 * shared_rho * np.sqrt(
        np.maximum(predicted_home_variance, 1e-6) * np.maximum(predicted_away_variance, 1e-6)
    )
    predicted_diff_variance = float(
        np.maximum(1e-6, predicted_home_variance + predicted_away_variance - covariance_adjustment)
    )
    return predicted_diff, predicted_diff_variance


def probability_total_over(
    predicted_home_corners: float,
    predicted_home_variance: float,
    predicted_away_corners: float,
    predicted_away_variance: float,
    quoted_line: float,
    shared_rho: float,
) -> float:
    """Compute `P(total corners > line)` from the Q1 moments."""
    predicted_total, predicted_total_variance = combined_total_moments(
        predicted_home_corners,
        predicted_home_variance,
        predicted_away_corners,
        predicted_away_variance,
        shared_rho,
    )
    max_corners = max(int(np.ceil(quoted_line)) + 20, int(predicted_total + 6 * np.sqrt(predicted_total_variance)) + 1)
    corner_grid = np.arange(0, max_corners + 1)
    total_size, total_prob = negative_binomial_params(predicted_total, predicted_total_variance)
    total_distribution = nbinom.pmf(corner_grid, total_size, total_prob)
    return float(np.clip(total_distribution[corner_grid > quoted_line].sum(), 0.0, 1.0))


def probability_diff_outcome(
    predicted_home_corners: float,
    predicted_home_variance: float,
    predicted_away_corners: float,
    predicted_away_variance: float,
    quoted_line: float,
    outcome_side: str,
    shared_rho: float,
) -> float:
    """Compute handicap or 1X2 outcome probabilities from the corner-difference moments."""
    predicted_diff, predicted_diff_variance = combined_diff_moments(
        predicted_home_corners,
        predicted_home_variance,
        predicted_away_corners,
        predicted_away_variance,
        shared_rho,
    )
    predicted_diff_std = float(np.maximum(np.sqrt(predicted_diff_variance), 1e-6))
    integer_line = np.isclose(quoted_line, round(quoted_line))

    if integer_line:
        lower_bound = quoted_line - 0.5
        upper_bound = quoted_line + 0.5
        if outcome_side == "home":
            return float(np.clip(1.0 - norm.cdf(upper_bound, loc=predicted_diff, scale=predicted_diff_std), 0.0, 1.0))
        if outcome_side == "away":
            return float(np.clip(norm.cdf(lower_bound, loc=predicted_diff, scale=predicted_diff_std), 0.0, 1.0))
        return float(
            np.clip(
                norm.cdf(upper_bound, loc=predicted_diff, scale=predicted_diff_std)
                - norm.cdf(lower_bound, loc=predicted_diff, scale=predicted_diff_std),
                0.0,
                1.0,
            )
        )

    if outcome_side == "home":
        return float(np.clip(1.0 - norm.cdf(quoted_line, loc=predicted_diff, scale=predicted_diff_std), 0.0, 1.0))
    if outcome_side == "away":
        return float(np.clip(norm.cdf(quoted_line, loc=predicted_diff, scale=predicted_diff_std), 0.0, 1.0))
    return 0.0


def quoted_market_line(price_row: pd.Series) -> float:
    """Read the OU or handicap line from the market row."""
    return float(price_row["od"]) if price_row["odds_type"] in {"OU", "HC"} else 0.0


def market_outcome_probabilities(prediction_row: pd.Series, price_row: pd.Series, shared_rho: float) -> dict[str, float]:
    """Turn one Q1 prediction row into market outcome probabilities."""
    predicted_home_corners = float(prediction_row["q2_home_center"])
    predicted_away_corners = float(prediction_row["q2_away_center"])
    predicted_home_variance = float(prediction_row["q2_home_variance"])
    predicted_away_variance = float(prediction_row["q2_away_variance"])
    market_type = price_row["odds_type"]
    market_line = quoted_market_line(price_row)

    if market_type == "OU":
        over_prob = probability_total_over(
            predicted_home_corners,
            predicted_home_variance,
            predicted_away_corners,
            predicted_away_variance,
            market_line,
            shared_rho,
        )
        return {"over": over_prob, "under": 1.0 - over_prob}

    if market_type == "HC":
        return {
            "home": probability_diff_outcome(
                predicted_home_corners,
                predicted_home_variance,
                predicted_away_corners,
                predicted_away_variance,
                market_line,
                "home",
                shared_rho,
            ),
            "push": probability_diff_outcome(
                predicted_home_corners,
                predicted_home_variance,
                predicted_away_corners,
                predicted_away_variance,
                market_line,
                "push",
                shared_rho,
            ),
            "away": probability_diff_outcome(
                predicted_home_corners,
                predicted_home_variance,
                predicted_away_corners,
                predicted_away_variance,
                market_line,
                "away",
                shared_rho,
            ),
        }

    return {
        "home": probability_diff_outcome(
            predicted_home_corners,
            predicted_home_variance,
            predicted_away_corners,
            predicted_away_variance,
            0.0,
            "home",
            shared_rho,
        ),
        "draw": probability_diff_outcome(
            predicted_home_corners,
            predicted_home_variance,
            predicted_away_corners,
            predicted_away_variance,
            0.0,
            "push",
            shared_rho,
        ),
        "away": probability_diff_outcome(
            predicted_home_corners,
            predicted_home_variance,
            predicted_away_corners,
            predicted_away_variance,
            0.0,
            "away",
            shared_rho,
        ),
    }


def build_candidate_side_rows(
    prediction_rows: pd.DataFrame,
    market_prices: pd.DataFrame,
    shared_rho: float,
    partition_label: str,
) -> pd.DataFrame:
    """Create one row per candidate betting side for one partition."""
    prediction_lookup = prediction_rows.set_index("match_id").to_dict("index")
    relevant_prices = market_prices[market_prices["match_id"].isin(prediction_rows["match_id"])].copy()
    candidate_rows: list[dict[str, object]] = []

    for _, price_row in relevant_prices.iterrows():
        match_id = int(price_row["match_id"])
        prediction_row = prediction_lookup.get(match_id)
        if prediction_row is None:
            continue

        market_type = price_row["odds_type"]
        if market_type in {"OU", "HC"} and pd.isna(price_row["od"]):
            continue
        market_line = quoted_market_line(price_row)
        quoted_probabilities = market_outcome_probabilities(prediction_row, price_row, shared_rho)
        market_group_id = f"{match_id}|{market_type}|{market_line}"
        quoted_odds = {
            "OU": {"over": price_row["oh"], "under": price_row["oa"]},
            "HC": {"home": price_row["oh"], "push": np.nan, "away": price_row["oa"]},
            "1X2": {"home": price_row["oh"], "draw": price_row["od"], "away": price_row["oa"]},
        }[market_type]

        for side_name, model_probability in quoted_probabilities.items():
            decimal_odds = quoted_odds.get(side_name, np.nan)
            candidate_rows.append(
                {
                    "match_id": match_id,
                    "date_time": pd.Timestamp(price_row["date_time"]),
                    "competition_id": price_row["competition_id"],
                    "partition": partition_label,
                    "odds_type": market_type,
                    "line": market_line,
                    "group_id": market_group_id,
                    "side": side_name,
                    "p_raw": float(np.clip(model_probability, 1e-6, 1 - 1e-6)),
                    "quantile_tail_distance": float(quantile_tail_distance(pd.Series(prediction_row), market_type, market_line, side_name)),
                    "odds": decimal_odds,
                    "bettable": bool(not np.isnan(decimal_odds)),
                }
            )

    return pd.DataFrame(candidate_rows)


def attach_observed_outcomes(candidate_bets: pd.DataFrame, observed_results: pd.DataFrame) -> pd.DataFrame:
    """Attach realised win/loss indicators to each candidate side."""
    merged = candidate_bets.merge(observed_results[["match_id", "home_corners", "away_corners"]], on="match_id", how="left")
    actual_total = merged["home_corners"] + merged["away_corners"]
    actual_diff = merged["home_corners"] - merged["away_corners"]
    won = np.full(len(merged), np.nan)

    mask_ou = merged["odds_type"] == "OU"
    won[mask_ou & (merged["side"] == "over")] = (
        actual_total[mask_ou & (merged["side"] == "over")] > merged.loc[mask_ou & (merged["side"] == "over"), "line"]
    ).astype(float)
    won[mask_ou & (merged["side"] == "under")] = (
        actual_total[mask_ou & (merged["side"] == "under")] < merged.loc[mask_ou & (merged["side"] == "under"), "line"]
    ).astype(float)

    mask_hc = merged["odds_type"] == "HC"
    won[mask_hc & (merged["side"] == "home")] = (
        actual_diff[mask_hc & (merged["side"] == "home")] > merged.loc[mask_hc & (merged["side"] == "home"), "line"]
    ).astype(float)
    won[mask_hc & (merged["side"] == "away")] = (
        actual_diff[mask_hc & (merged["side"] == "away")] < merged.loc[mask_hc & (merged["side"] == "away"), "line"]
    ).astype(float)
    won[mask_hc & (merged["side"] == "push")] = (
        actual_diff[mask_hc & (merged["side"] == "push")] == merged.loc[mask_hc & (merged["side"] == "push"), "line"]
    ).astype(float)

    mask_1x2 = merged["odds_type"] == "1X2"
    won[mask_1x2 & (merged["side"] == "home")] = (actual_diff[mask_1x2 & (merged["side"] == "home")] > 0).astype(float)
    won[mask_1x2 & (merged["side"] == "draw")] = (actual_diff[mask_1x2 & (merged["side"] == "draw")] == 0).astype(float)
    won[mask_1x2 & (merged["side"] == "away")] = (actual_diff[mask_1x2 & (merged["side"] == "away")] < 0).astype(float)

    merged["won"] = won
    return merged.dropna(subset=["won"]).copy()


def brier_score(actual: pd.Series | np.ndarray, predicted_probability: pd.Series | np.ndarray) -> float:
    """Compute mean Brier score."""
    actual_array = np.asarray(actual, dtype=float)
    probability_array = np.asarray(predicted_probability, dtype=float)
    return float(np.mean((actual_array - probability_array) ** 2))


def log_loss_score(actual: pd.Series | np.ndarray, predicted_probability: pd.Series | np.ndarray) -> float:
    """Compute binary log loss with safe clipping."""
    actual_array = np.asarray(actual, dtype=float)
    probability_array = np.clip(np.asarray(predicted_probability, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(actual_array * np.log(probability_array) + (1 - actual_array) * np.log(1 - probability_array)))


def fit_isotonic_side_calibrators(training_bets: pd.DataFrame) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    """Fit per-market-side isotonic calibrators on partition A."""
    calibrators: dict[str, dict[str, object]] = {}
    report_rows: list[dict[str, object]] = []
    training_rows = training_bets.dropna(subset=["p_raw", "won"]).copy()
    training_rows["calibration_key"] = training_rows["odds_type"] + "|" + training_rows["side"]

    for calibration_key, group in training_rows.groupby("calibration_key"):
        p_raw = group["p_raw"].to_numpy(dtype=float)
        won = group["won"].to_numpy(dtype=float)
        enough_data = len(group) >= CALIBRATION_MIN_SAMPLES and np.unique(p_raw).size >= 20

        if enough_data:
            isotonic_model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            isotonic_model.fit(p_raw, won)
            shrink_weight = float(len(group) / (len(group) + CALIBRATION_SHRINKAGE))
            isotonic_fit = np.asarray(isotonic_model.predict(p_raw), dtype=float)
            calibrated_probability = np.clip(p_raw + shrink_weight * (isotonic_fit - p_raw), 1e-6, 1 - 1e-6)
            method_name = "isotonic_shrink"
        else:
            isotonic_model = None
            shrink_weight = 0.0
            calibrated_probability = np.clip(p_raw, 1e-6, 1 - 1e-6)
            method_name = "identity"

        calibrators[calibration_key] = {"model": isotonic_model, "shrink_weight": shrink_weight, "method": method_name}
        report_rows.append(
            {
                "odds_type": group["odds_type"].iloc[0],
                "side": group["side"].iloc[0],
                "n_train": len(group),
                "method": method_name,
                "alpha": shrink_weight,
                "win_rate": float(won.mean()),
                "raw_brier": brier_score(won, p_raw),
                "cal_brier": brier_score(won, calibrated_probability),
                "raw_logloss": log_loss_score(won, p_raw),
                "cal_logloss": log_loss_score(won, calibrated_probability),
            }
        )

    report = pd.DataFrame(report_rows).sort_values(["odds_type", "side"]).reset_index(drop=True)
    return calibrators, report


def apply_base_probability_calibration(candidate_bets: pd.DataFrame, calibrators: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Apply the per-side isotonic calibration and renormalize within each market."""
    calibrated = candidate_bets.copy()
    calibrated["calibration_key"] = calibrated["odds_type"] + "|" + calibrated["side"]
    pre_normalized_probabilities: list[float] = []

    for _, row in calibrated.iterrows():
        raw_probability = float(np.clip(row["p_raw"], 1e-6, 1 - 1e-6))
        calibrator = calibrators.get(row["calibration_key"])
        if calibrator is None or calibrator["method"] == "identity":
            pre_normalized_probabilities.append(raw_probability)
            continue
        isotonic_probability = float(calibrator["model"].predict([raw_probability])[0])
        shrink_weight = float(calibrator["shrink_weight"])
        pre_normalized_probabilities.append(
            float(np.clip(raw_probability + shrink_weight * (isotonic_probability - raw_probability), 1e-6, 1 - 1e-6))
        )

    calibrated["p_pre_cal"] = pre_normalized_probabilities
    probability_sum = calibrated.groupby("group_id")["p_pre_cal"].transform("sum").clip(lower=1e-6)
    calibrated["p_model"] = np.clip(calibrated["p_pre_cal"] / probability_sum, 1e-6, 1 - 1e-6)
    calibrated["ev_raw"] = np.where(calibrated["bettable"], calibrated["p_raw"] * calibrated["odds"] - 1, np.nan)
    calibrated["ev"] = np.where(calibrated["bettable"], calibrated["p_model"] * calibrated["odds"] - 1, np.nan)
    return calibrated


def attach_no_vig_market_probabilities(candidate_bets: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal odds inside each market into no-vig reference probabilities."""
    with_market_probs = candidate_bets.copy()
    with_market_probs["inv_odds"] = np.where(with_market_probs["bettable"], 1.0 / with_market_probs["odds"], np.nan)
    probability_sum = with_market_probs.groupby("group_id")["inv_odds"].transform("sum")
    with_market_probs["p_market"] = np.where(with_market_probs["bettable"], with_market_probs["inv_odds"] / probability_sum, np.nan)
    return with_market_probs


def resolve_market_thresholds(
    ev_threshold: float | None = None,
    market_thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    """Resolve the effective per-market EV thresholds for bet selection."""
    if market_thresholds is not None:
        return {market_name: float(value) for market_name, value in market_thresholds.items()}
    if ev_threshold is None:
        return dict(DEFAULT_MARKET_THRESHOLDS)
    return {market_name: float(ev_threshold) for market_name in ["1X2", "HC", "OU"]}


def threshold_column_name(probability_column: str) -> str:
    """Convert an EV column name into the corresponding threshold column name."""
    return "threshold_" + probability_column.replace("ev_", "")


def attach_market_thresholds(candidate_bets: pd.DataFrame, market_thresholds: dict[str, float]) -> pd.DataFrame:
    """Attach market-specific threshold columns to one bet table."""
    with_thresholds = candidate_bets.copy()
    with_thresholds["threshold_ev"] = with_thresholds["odds_type"].map(market_thresholds).fillna(DEFAULT_EV_THRESHOLD)
    for probability_column in ["ev_raw", "ev", "ev_tail"]:
        if probability_column in with_thresholds.columns:
            with_thresholds[threshold_column_name(probability_column)] = with_thresholds["threshold_ev"]
    return with_thresholds


def select_bets_with_market_thresholds(
    candidate_bets: pd.DataFrame,
    ev_column: str,
) -> pd.DataFrame:
    """Select bettable sides whose EV clears the threshold assigned to their market."""
    threshold_col = threshold_column_name(ev_column)
    selected = candidate_bets[
        candidate_bets["bettable"] & candidate_bets[ev_column].notna() & (candidate_bets[ev_column] > candidate_bets[threshold_col])
    ].copy()
    return selected


def apply_tail_probability_shrink(
    candidate_bets: pd.DataFrame,
    lambda_by_market: dict[str, float],
    tail_scale: float,
) -> pd.DataFrame:
    """Shrink only the positive-EV tail back toward no-vig market probabilities."""
    shrunk = candidate_bets.copy()
    shrunk["p_tail"] = shrunk["p_model"]
    shrunk["ev_tail"] = shrunk["ev"]
    shrunk["tail_weight"] = 0.0

    mask = shrunk["bettable"] & shrunk["odds_type"].isin(TAIL_SHRINK_MARKETS)
    if not mask.any():
        return shrunk

    subset = shrunk.loc[mask].copy()
    lambda_values = subset["odds_type"].map(lambda_by_market).fillna(0.0).to_numpy(dtype=float)
    positive_edge = np.maximum(subset["ev"].to_numpy(dtype=float) - DEFAULT_EV_THRESHOLD, 0.0)
    quantile_multiplier = 1.0 + subset["quantile_tail_distance"].fillna(0.0).to_numpy(dtype=float)
    tail_weight = 1.0 - np.exp(-(lambda_values * tail_scale) * positive_edge * quantile_multiplier)
    tail_probability = np.clip(
        (1.0 - tail_weight) * subset["p_model"].to_numpy(dtype=float) + tail_weight * subset["p_market"].to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )

    shrunk.loc[mask, "tail_weight"] = tail_weight
    shrunk.loc[mask, "p_tail"] = tail_probability
    shrunk.loc[mask, "ev_tail"] = shrunk.loc[mask, "p_tail"] * shrunk.loc[mask, "odds"] - 1.0
    return shrunk


def fit_tail_lambda(training_bets: pd.DataFrame, odds_type: str, market_threshold: float) -> float:
    """Choose how aggressively one market's high-EV tail should revert to market probabilities."""
    subset = training_bets[(training_bets["bettable"]) & (training_bets["odds_type"] == odds_type) & (training_bets["ev"] > market_threshold)].copy()
    if subset.empty:
        return 0.0

    actual_wins = subset["won"].to_numpy(dtype=float)
    model_probability = subset["p_model"].to_numpy(dtype=float)
    market_probability = subset["p_market"].to_numpy(dtype=float)
    positive_edge = np.maximum(subset["ev"].to_numpy(dtype=float) - market_threshold, 0.0)

    def objective(lambda_value: float) -> float:
        tail_weight = 1.0 - np.exp(-lambda_value * positive_edge)
        tail_probability = np.clip(
            (1.0 - tail_weight) * model_probability + tail_weight * market_probability,
            1e-6,
            1 - 1e-6,
        )
        return float(-np.mean(actual_wins * np.log(tail_probability) + (1.0 - actual_wins) * np.log(1.0 - tail_probability)))

    result = minimize_scalar(objective, bounds=(0.0, TAIL_SHRINK_MAX_LAMBDA), method="bounded")
    return float(result.x)


def realised_roi(selected_bets: pd.DataFrame) -> float:
    """Compute mean per-bet realised profit."""
    if selected_bets.empty:
        return np.nan
    profit = np.where(selected_bets["won"] == 1, STAKE * (selected_bets["odds"] - 1.0), -STAKE)
    return float(np.mean(profit))


def choose_tail_scale(
    training_bets: pd.DataFrame,
    lambda_by_market: dict[str, float],
    market_thresholds: dict[str, float],
) -> tuple[float, pd.DataFrame]:
    """Choose the tail-shrink scale that best aligns A-side mean EV with realised ROI."""
    diagnostics: list[dict[str, float | int]] = []
    best_scale = 0.0
    best_gap = np.inf

    for candidate_scale in TAIL_SCALE_GRID:
        shrunk = apply_tail_probability_shrink(training_bets, lambda_by_market, candidate_scale)
        shrunk = attach_market_thresholds(shrunk, market_thresholds)
        selected = select_bets_with_market_thresholds(shrunk, "ev_tail")
        mean_ev = float(selected["ev_tail"].mean()) if not selected.empty else np.nan
        actual_roi = realised_roi(selected)
        gap = abs(mean_ev - actual_roi) if np.isfinite(mean_ev) and np.isfinite(actual_roi) else np.inf
        diagnostics.append({"tail_scale": candidate_scale, "bets": len(selected), "mean_ev": mean_ev, "roi": actual_roi, "gap": gap})
        if gap < best_gap:
            best_gap = gap
            best_scale = candidate_scale

    return float(best_scale), pd.DataFrame(diagnostics)


def summarize_market_calibration(raw_bets: pd.DataFrame, calibrated_bets: pd.DataFrame, partition_label: str) -> pd.DataFrame:
    """Summarize Brier, log loss, and average EV by market."""
    rows: list[dict[str, object]] = []
    for odds_type in sorted(calibrated_bets["odds_type"].unique()):
        raw_market = raw_bets[raw_bets["odds_type"] == odds_type]
        calibrated_market = calibrated_bets[calibrated_bets["odds_type"] == odds_type]
        raw_ev = raw_market.loc[raw_market["bettable"], "p_raw"] * raw_market.loc[raw_market["bettable"], "odds"] - 1
        rows.append(
            {
                "partition": partition_label,
                "odds_type": odds_type,
                "n_sides": len(calibrated_market),
                "bettable_sides": int(calibrated_market["bettable"].sum()),
                "raw_brier": brier_score(raw_market["won"], raw_market["p_raw"]),
                "cal_brier": brier_score(calibrated_market["won"], calibrated_market["p_model"]),
                "raw_logloss": log_loss_score(raw_market["won"], raw_market["p_raw"]),
                "cal_logloss": log_loss_score(calibrated_market["won"], calibrated_market["p_model"]),
                "mean_raw_ev": float(raw_ev.mean()),
                "mean_cal_ev": float(calibrated_market.loc[calibrated_market["bettable"], "ev"].mean()),
            }
        )
    return pd.DataFrame(rows)


def full_kelly_fraction(win_probability: np.ndarray, decimal_odds: np.ndarray) -> np.ndarray:
    """Compute a practical full-Kelly bankroll fraction for decimal odds, capped per bet."""
    net_odds = np.maximum(decimal_odds - 1.0, 1e-6)
    raw_fraction = (win_probability * decimal_odds - 1.0) / net_odds
    return np.clip(raw_fraction, 0.0, MAX_FULL_KELLY_FRACTION)


def run_staking_path(
    selected_bets: pd.DataFrame,
    stake_mode: str,
    fraction_scale: float,
    initial_bankroll: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replay one realised bet sequence under fixed-stake or fractional Kelly sizing."""
    path = selected_bets.sort_values("date_time").reset_index(drop=True).copy()
    bankroll = float(initial_bankroll)
    bankroll_before: list[float] = []
    stake_sizes: list[float] = []
    pnl_values: list[float] = []
    bankroll_after: list[float] = []

    full_fraction = full_kelly_fraction(path["p_tail"].to_numpy(dtype=float), path["odds"].to_numpy(dtype=float))
    path["full_kelly_fraction"] = full_fraction
    if stake_mode == "fixed":
        target_fraction = np.zeros(len(path), dtype=float)
    else:
        target_fraction = np.clip(full_fraction * fraction_scale, 0.0, 1.0)
    path["stake_fraction"] = target_fraction

    for _, group in path.groupby("date_time", sort=True):
        bankroll_group_start = bankroll
        group_indices = group.index.to_list()
        if stake_mode == "fixed":
            group_stakes = np.repeat(STAKE * fraction_scale, len(group_indices)).astype(float)
            total_group_stake = group_stakes.sum()
            if total_group_stake > bankroll_group_start and total_group_stake > 0:
                group_stakes *= bankroll_group_start / total_group_stake
        else:
            group_stakes = bankroll_group_start * target_fraction[group_indices]
            total_group_stake = group_stakes.sum()
            if total_group_stake > bankroll_group_start and total_group_stake > 0:
                group_stakes *= bankroll_group_start / total_group_stake

        group_profits = np.where(
            group["won"].to_numpy(dtype=float) == 1.0,
            group_stakes * (group["odds"].to_numpy(dtype=float) - 1.0),
            -group_stakes,
        )
        bankroll += float(group_profits.sum())

        for local_pos, group_index in enumerate(group_indices):
            bankroll_before.append(bankroll_group_start)
            stake_sizes.append(float(group_stakes[local_pos]))
            pnl_values.append(float(group_profits[local_pos]))
            bankroll_after.append(bankroll_group_start + float(np.sum(group_profits[: local_pos + 1])))

    path["bankroll_before"] = bankroll_before
    path["stake_size"] = stake_sizes
    path["strategy_pnl"] = pnl_values
    path["bankroll_after"] = bankroll_after
    peak_bankroll = path["bankroll_after"].cummax() if len(path) else pd.Series(dtype=float)
    drawdown = (path["bankroll_after"] - peak_bankroll) / peak_bankroll if len(path) else pd.Series(dtype=float)

    total_staked = float(path["stake_size"].sum())
    total_pnl = float(path["strategy_pnl"].sum())
    ending_bankroll = float(bankroll)
    summary = {
        "bets": int(len(path)),
        "total_staked": total_staked,
        "total_pnl": total_pnl,
        "turnover_roi": total_pnl / total_staked if total_staked else np.nan,
        "ending_bankroll": ending_bankroll,
        "bankroll_return": ending_bankroll / initial_bankroll - 1.0 if initial_bankroll else np.nan,
        "max_drawdown": float(drawdown.min()) if len(path) else np.nan,
        "average_stake": float(path["stake_size"].mean()) if len(path) else np.nan,
        "median_stake": float(path["stake_size"].median()) if len(path) else np.nan,
        "average_fraction": float(path["stake_fraction"].mean()) if len(path) else np.nan,
    }
    return path, summary


def simulate_staking_distribution(
    selected_bets: pd.DataFrame,
    stake_mode: str,
    fraction_scale: float,
    initial_bankroll: float,
    monte_carlo_runs: int,
) -> dict[str, float]:
    """Simulate final PnL and ending bankroll distribution under one staking plan."""
    if selected_bets.empty:
        return {
            "mc_p5_pnl": np.nan,
            "mc_p50_pnl": np.nan,
            "mc_p95_pnl": np.nan,
            "mc_p5_bankroll": np.nan,
            "mc_p50_bankroll": np.nan,
            "mc_p95_bankroll": np.nan,
            "mc_positive_prob": np.nan,
            "mc_actual_percentile": np.nan,
            "simulated_total_pnl": np.array([], dtype=float),
        }

    rng = np.random.default_rng(42)
    win_probability = selected_bets["p_tail"].to_numpy(dtype=float)
    decimal_odds = selected_bets["odds"].to_numpy(dtype=float)
    realised_wins = selected_bets["won"].to_numpy(dtype=float)
    simulated_wins = rng.random((monte_carlo_runs, len(selected_bets))) < win_probability[None, :]
    bankrolls = np.full(monte_carlo_runs, float(initial_bankroll), dtype=float)

    full_fraction = full_kelly_fraction(win_probability, decimal_odds)
    if stake_mode == "fixed":
        per_bet_fraction = np.zeros(len(selected_bets), dtype=float)
    else:
        per_bet_fraction = np.clip(full_fraction * fraction_scale, 0.0, 1.0)

    actual_path, actual_summary = run_staking_path(selected_bets, stake_mode, fraction_scale, initial_bankroll)
    del actual_path

    ordered_bets = selected_bets.sort_values("date_time").reset_index(drop=True)
    date_groups = ordered_bets.groupby("date_time", sort=True).indices
    for group_indices in date_groups.values():
        group_indices = np.asarray(group_indices, dtype=int)
        group_bankroll_start = bankrolls.copy()
        if stake_mode == "fixed":
            group_stakes = np.full((monte_carlo_runs, len(group_indices)), STAKE * fraction_scale, dtype=float)
            stake_totals = group_stakes.sum(axis=1)
            scale = np.where(stake_totals > group_bankroll_start, group_bankroll_start / np.maximum(stake_totals, 1e-12), 1.0)
            group_stakes = group_stakes * scale[:, None]
        else:
            group_stakes = group_bankroll_start[:, None] * per_bet_fraction[group_indices][None, :]
            stake_totals = group_stakes.sum(axis=1)
            scale = np.where(stake_totals > group_bankroll_start, group_bankroll_start / np.maximum(stake_totals, 1e-12), 1.0)
            group_stakes = group_stakes * scale[:, None]

        group_odds = decimal_odds[group_indices][None, :]
        group_wins = simulated_wins[:, group_indices]
        group_profits = np.where(group_wins, group_stakes * (group_odds - 1.0), -group_stakes)
        bankrolls = bankrolls + group_profits.sum(axis=1)

    simulated_total_pnl = bankrolls - initial_bankroll
    return {
        "mc_p5_pnl": float(np.percentile(simulated_total_pnl, 5)),
        "mc_p50_pnl": float(np.percentile(simulated_total_pnl, 50)),
        "mc_p95_pnl": float(np.percentile(simulated_total_pnl, 95)),
        "mc_p5_bankroll": float(np.percentile(bankrolls, 5)),
        "mc_p50_bankroll": float(np.percentile(bankrolls, 50)),
        "mc_p95_bankroll": float(np.percentile(bankrolls, 95)),
        "mc_positive_prob": float((simulated_total_pnl > 0).mean()),
        "mc_actual_percentile": float((simulated_total_pnl <= actual_summary["total_pnl"]).mean() * 100),
        "simulated_total_pnl": simulated_total_pnl,
    }


def compare_staking_plans(
    selected_bets: pd.DataFrame,
    monte_carlo_runs: int,
    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compare baseline fixed stakes with multiple Kelly fractions on the same selected bets."""
    bet_paths: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | str]] = []

    for strategy_name, config in DEFAULT_STAKING_PLAN.items():
        path, realised_summary = run_staking_path(
            selected_bets,
            stake_mode=str(config["mode"]),
            fraction_scale=float(config["fraction"]),
            initial_bankroll=initial_bankroll,
        )
        mc_summary = simulate_staking_distribution(
            selected_bets,
            stake_mode=str(config["mode"]),
            fraction_scale=float(config["fraction"]),
            initial_bankroll=initial_bankroll,
            monte_carlo_runs=monte_carlo_runs,
        )
        bet_paths[strategy_name] = path
        summary_rows.append(
            {
                "strategy": strategy_name,
                "stake_mode": str(config["mode"]),
                "fraction_scale": float(config["fraction"]),
                **realised_summary,
                **{key: value for key, value in mc_summary.items() if key != "simulated_total_pnl"},
            }
        )

    summary = pd.DataFrame(summary_rows)
    return bet_paths, summary


def evaluate_selected_bets(selected_bets: pd.DataFrame, monte_carlo_runs: int) -> dict[str, float]:
    """Compute realised PnL, ROI, and Monte Carlo diagnostics for selected bets."""
    selected = selected_bets.copy()
    selected["pnl"] = np.where(selected["won"] == 1, STAKE * (selected["odds"] - 1), -STAKE)
    total_pnl = float(selected["pnl"].sum())
    total_staked = float(len(selected) * STAKE)
    roi = total_pnl / total_staked if total_staked else np.nan

    rng = np.random.default_rng(42)
    win_probability = selected["p_tail"].to_numpy(dtype=float)
    decimal_odds = selected["odds"].to_numpy(dtype=float)
    simulated_wins = rng.random((monte_carlo_runs, len(selected))) < win_probability[None, :]
    simulated_pnl = np.where(simulated_wins, STAKE * (decimal_odds[None, :] - 1), -STAKE)
    simulated_total_pnl = simulated_pnl.sum(axis=1)

    return {
        "bets": len(selected),
        "total_pnl": total_pnl,
        "roi": roi,
        "win_rate": float(selected["won"].mean()),
        "mean_tail_ev": float(selected["ev_tail"].mean()),
        "mean_base_ev": float(selected["ev"].mean()),
        "mean_raw_ev": float(selected["ev_raw"].mean()),
        "mc_p5": float(np.percentile(simulated_total_pnl, 5)),
        "mc_p50": float(np.percentile(simulated_total_pnl, 50)),
        "mc_p95": float(np.percentile(simulated_total_pnl, 95)),
        "mc_positive_prob": float((simulated_total_pnl > 0).mean()),
        "mc_actual_percentile": float((simulated_total_pnl <= total_pnl).mean() * 100),
        "simulated_total_pnl": simulated_total_pnl,
    }


def plot_cumulative_pnl(selected_bets: pd.DataFrame, output_dir: Path) -> None:
    """Plot cumulative realised vs expected PnL."""
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.plot(selected_bets["date_time"], selected_bets["cum_pnl"], label="Realised PnL", linewidth=1.5)
    axis.plot(
        selected_bets["date_time"],
        selected_bets["cum_ev"],
        label="Expected PnL (cumulative calibrated EV)",
        linewidth=1.5,
        linestyle="--",
        color="orange",
    )
    axis.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    axis.set_title("Cumulative PnL vs Expected PnL over time")
    axis.set_xlabel("Date")
    axis.set_ylabel("Units")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q2_cumulative_pnl.png", dpi=150)
    plt.close(fig)


def plot_rolling_pnl(selected_bets: pd.DataFrame, output_dir: Path) -> None:
    """Plot realised and expected PnL in 14-day bins."""
    resampled = selected_bets.copy()
    resampled["date_time"] = pd.to_datetime(resampled["date_time"])
    resampled = resampled.set_index("date_time").sort_index()
    rolling_pnl = resampled["pnl"].resample("14D").sum()
    rolling_ev = (resampled["ev_tail"] * STAKE).resample("14D").sum()

    fig, axis = plt.subplots(figsize=(11, 4))
    axis.bar(rolling_pnl.index, rolling_pnl.values, width=12, alpha=0.6, label="Realised PnL (14D bin)")
    axis.step(
        rolling_ev.index,
        rolling_ev.values,
        where="mid",
        color="orange",
        linewidth=1.8,
        linestyle="--",
        label="Expected PnL (14D bin)",
    )
    axis.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    axis.set_title("14-day rolling PnL and expected PnL")
    axis.set_xlabel("Date")
    axis.set_ylabel("Units")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q2_rolling_pnl.png", dpi=150)
    plt.close(fig)


def plot_monte_carlo_distribution(evaluation: dict[str, float], output_dir: Path) -> None:
    """Plot the Monte Carlo distribution of total PnL."""
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.hist(
        evaluation["simulated_total_pnl"],
        bins=80,
        density=True,
        alpha=0.7,
        color="steelblue",
        label="MC distribution",
    )
    axis.axvline(evaluation["total_pnl"], color="red", linewidth=2, label=f"Actual PnL = {evaluation['total_pnl']:+.2f}")
    axis.axvline(
        evaluation["mc_p50"],
        color="orange",
        linewidth=1.5,
        linestyle="--",
        label=f"MC median = {evaluation['mc_p50']:+.2f}",
    )
    axis.axvline(0, color="grey", linewidth=0.8, linestyle=":")
    axis.set_title("Monte Carlo distribution of total PnL")
    axis.set_xlabel("Total PnL (units)")
    axis.set_ylabel("Density")
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q2_monte_carlo.png", dpi=150)
    plt.close(fig)


def save_q2_outputs(
    output_dir: Path,
    partition_a_bets: pd.DataFrame,
    partition_b_bets: pd.DataFrame,
    selected_bets: pd.DataFrame,
    staking_paths: dict[str, pd.DataFrame],
    base_calibration_report: pd.DataFrame,
    partition_summary: pd.DataFrame,
    tail_scale_report: pd.DataFrame,
    run_summary: pd.DataFrame,
    staking_summary: pd.DataFrame,
) -> None:
    """Persist Q2 tables for one run into the chosen output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def safe_write_csv(frame: pd.DataFrame, path: Path) -> None:
        try:
            frame.to_csv(path, index=False)
        except PermissionError:
            fallback = path.with_name(f"{path.stem}.latest{path.suffix}")
            frame.to_csv(fallback, index=False)

    safe_write_csv(partition_a_bets, output_dir / "partition_a_bets.csv")
    safe_write_csv(partition_b_bets, output_dir / "partition_b_bets.csv")
    safe_write_csv(selected_bets, output_dir / "selected_bets.csv")
    for strategy_name, path in staking_paths.items():
        safe_write_csv(path, output_dir / f"{strategy_name}_bet_path.csv")
    safe_write_csv(base_calibration_report, output_dir / "base_calibration_report.csv")
    safe_write_csv(partition_summary, output_dir / "partition_summary.csv")
    safe_write_csv(tail_scale_report, output_dir / "tail_scale_report.csv")
    safe_write_csv(run_summary, output_dir / "run_summary.csv")
    safe_write_csv(staking_summary, output_dir / "staking_summary.csv")


def run_q2_pipeline(
    data_dir: str | Path = ".",
    ev_threshold: float | None = None,
    market_thresholds: dict[str, float] | None = None,
    monte_carlo_runs: int = DEFAULT_MONTE_CARLO_RUNS,
    output_dir: str | Path | None = None,
    evaluation_scope: str = "B",
    make_plots: bool = True,
) -> Q2Artifacts:
    """Run the full Q2 market-probability, calibration, and backtest pipeline."""
    data_path = Path(data_dir)
    result_path = Path(output_dir) if output_dir is not None else data_path
    result_path.mkdir(parents=True, exist_ok=True)
    effective_thresholds = resolve_market_thresholds(ev_threshold, market_thresholds)
    predicted_matches, market_prices, observed_results = read_prediction_inputs(data_path)
    predicted_matches = attach_quantile_aware_distribution_inputs(predicted_matches)
    partition_a_predictions = predicted_matches[predicted_matches["partition"] == "A"].copy()
    partition_b_predictions = predicted_matches[predicted_matches["partition"] == "B"].copy()

    shared_rho_info = estimate_shared_correlation(partition_a_predictions)
    print_heading("Shared Correlation")
    print(
        f"rho={shared_rho_info['rho']:.3f} from A only | "
        f"n={shared_rho_info['n']} | "
        f"k_total={shared_rho_info['k_total']:.3f} | "
        f"k_diff={shared_rho_info['k_diff']:.3f}"
    )

    partition_a_raw = attach_observed_outcomes(
        build_candidate_side_rows(partition_a_predictions, market_prices, shared_rho_info["rho"], "A"),
        observed_results,
    )
    partition_b_raw = attach_observed_outcomes(
        build_candidate_side_rows(partition_b_predictions, market_prices, shared_rho_info["rho"], "B"),
        observed_results,
    )

    print_heading("Candidate Sides")
    print(
        f"A={len(partition_a_raw)} sides {partition_a_raw['odds_type'].value_counts().to_dict()} | "
        f"B={len(partition_b_raw)} sides {partition_b_raw['odds_type'].value_counts().to_dict()}"
    )

    base_calibrators, base_calibration_report = fit_isotonic_side_calibrators(partition_a_raw)
    partition_a_calibrated = apply_base_probability_calibration(partition_a_raw, base_calibrators)
    partition_b_calibrated = apply_base_probability_calibration(partition_b_raw, base_calibrators)
    partition_a_calibrated = attach_no_vig_market_probabilities(partition_a_calibrated)
    partition_b_calibrated = attach_no_vig_market_probabilities(partition_b_calibrated)

    partition_summary = pd.concat(
        [
            summarize_market_calibration(partition_a_raw, partition_a_calibrated, "A"),
            summarize_market_calibration(partition_b_raw, partition_b_calibrated, "B"),
        ],
        ignore_index=True,
    )

    tail_lambda_map = {
        market_name: fit_tail_lambda(partition_a_calibrated, market_name, effective_thresholds[market_name])
        for market_name in TAIL_SHRINK_MARKETS
    }
    tail_scale, tail_scale_report = choose_tail_scale(partition_a_calibrated, tail_lambda_map, effective_thresholds)
    partition_a_final = apply_tail_probability_shrink(partition_a_calibrated, tail_lambda_map, tail_scale)
    partition_b_final = apply_tail_probability_shrink(partition_b_calibrated, tail_lambda_map, tail_scale)
    partition_a_final = attach_market_thresholds(partition_a_final, effective_thresholds)
    partition_b_final = attach_market_thresholds(partition_b_final, effective_thresholds)

    print_heading("Base Calibration")
    print(base_calibration_report.round(4).to_string(index=False))
    print_heading("Partition Summary")
    print(partition_summary.round(4).to_string(index=False))
    print_heading("Tail Shrink")
    print(f"lambda map={tail_lambda_map}")
    print(f"selected tail scale={tail_scale:.2f}")
    print(f"market thresholds={effective_thresholds}")
    print(tail_scale_report.round(4).to_string(index=False))

    selected_a_bets = select_bets_with_market_thresholds(partition_a_final, "ev_tail")
    print(
        f"A selected tail check: bets={len(selected_a_bets)} "
        f"mean_ev={selected_a_bets['ev_tail'].mean():+.4f} "
        f"realised_roi={realised_roi(selected_a_bets):+.4f}"
    )

    if evaluation_scope.upper() == "ALL":
        evaluation_pool = pd.concat([partition_a_final, partition_b_final], ignore_index=True)
    else:
        evaluation_pool = partition_b_final.copy()

    selected_bets = select_bets_with_market_thresholds(evaluation_pool, "ev_tail")
    selected_bets = selected_bets.sort_values("date_time").reset_index(drop=True)
    selected_bets["pnl"] = np.where(selected_bets["won"] == 1, STAKE * (selected_bets["odds"] - 1), -STAKE)
    selected_bets["cum_pnl"] = selected_bets["pnl"].cumsum()
    selected_bets["cum_ev"] = (selected_bets["ev_tail"] * STAKE).cumsum()

    evaluation = evaluate_selected_bets(selected_bets, monte_carlo_runs)
    staking_paths, staking_summary = compare_staking_plans(selected_bets, monte_carlo_runs)
    run_summary = pd.DataFrame(
        [
            {
                "evaluation_scope": evaluation_scope.upper(),
                "threshold_1x2": effective_thresholds["1X2"],
                "threshold_hc": effective_thresholds["HC"],
                "threshold_ou": effective_thresholds["OU"],
                "bets": evaluation["bets"],
                "total_pnl": evaluation["total_pnl"],
                "roi": evaluation["roi"],
                "win_rate": evaluation["win_rate"],
                "mean_tail_ev": evaluation["mean_tail_ev"],
                "mean_base_ev": evaluation["mean_base_ev"],
                "mean_raw_ev": evaluation["mean_raw_ev"],
                "mc_p5": evaluation["mc_p5"],
                "mc_p50": evaluation["mc_p50"],
                "mc_p95": evaluation["mc_p95"],
                "mc_positive_prob": evaluation["mc_positive_prob"],
                "mc_actual_percentile": evaluation["mc_actual_percentile"],
            }
        ]
    )
    print_heading("Q2 Selection Summary")
    print(f"Bets placed        : {evaluation['bets']}")
    print(f"Evaluation scope   : {evaluation_scope.upper()}")
    print(f"EV thresholds      : 1X2>{effective_thresholds['1X2']:.3f} HC>{effective_thresholds['HC']:.3f} OU>{effective_thresholds['OU']:.3f}")
    print(f"Breakdown by market: {selected_bets['odds_type'].value_counts().to_dict()}")
    print_heading("Q2 Overall PnL / RoI")
    print(f"Total staked       : {evaluation['bets'] * STAKE:.2f} units")
    print(f"Total PnL          : {evaluation['total_pnl']:+.2f} units")
    print(f"RoI                : {evaluation['roi']:+.2%}")
    print(f"Win rate           : {evaluation['win_rate']:.2%}")
    print(f"Mean calibrated EV : {evaluation['mean_tail_ev']:+.4f}")
    print(f"Mean pre-tail EV   : {evaluation['mean_base_ev']:+.4f}")
    print(f"Mean raw EV        : {evaluation['mean_raw_ev']:+.4f}")
    print_heading("Monte Carlo")
    print(f"Actual total PnL   : {evaluation['total_pnl']:+.2f}")
    print(f"MC median PnL      : {evaluation['mc_p50']:+.2f}")
    print(f"MC 5th-95th pct    : [{evaluation['mc_p5']:+.2f}, {evaluation['mc_p95']:+.2f}]")
    print(f"P(positive PnL)    : {evaluation['mc_positive_prob']:.2%}")
    print(f"Actual percentile  : {evaluation['mc_actual_percentile']:.0f}th")
    print_heading("Staking Comparison")
    print(f"Kelly cap per bet : {MAX_FULL_KELLY_FRACTION:.1%} of bankroll")
    print(
        staking_summary[
            [
                "strategy",
                "total_pnl",
                "turnover_roi",
                "ending_bankroll",
                "bankroll_return",
                "max_drawdown",
                "mc_p50_pnl",
                "mc_actual_percentile",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    if make_plots:
        plot_cumulative_pnl(selected_bets, result_path)
        plot_rolling_pnl(selected_bets, result_path)
        plot_monte_carlo_distribution(evaluation, result_path)

    save_q2_outputs(
        result_path,
        partition_a_final,
        partition_b_final,
        selected_bets,
        staking_paths,
        base_calibration_report,
        partition_summary,
        tail_scale_report,
        run_summary,
        staking_summary,
    )

    return Q2Artifacts(
        partition_a_bets=partition_a_final,
        partition_b_bets=partition_b_final,
        selected_bets=selected_bets,
        base_calibration_report=base_calibration_report,
        partition_summary=partition_summary,
        tail_scale_report=tail_scale_report,
        run_summary=run_summary,
        staking_summary=staking_summary,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the standalone Q2 runner."""
    parser = argparse.ArgumentParser(description="Run the Q2 calibration and betting backtest pipeline.")
    parser.add_argument("--data-dir", default=".", help="Directory containing Q1 outputs and market price files.")
    parser.add_argument(
        "--ev-threshold",
        type=float,
        default=None,
        help="Optional single EV threshold applied to every market. Omit to use the per-market defaults.",
    )
    parser.add_argument("--threshold-1x2", type=float, default=DEFAULT_MARKET_THRESHOLDS["1X2"], help="EV threshold for 1X2 bets.")
    parser.add_argument("--threshold-hc", type=float, default=DEFAULT_MARKET_THRESHOLDS["HC"], help="EV threshold for handicap bets.")
    parser.add_argument("--threshold-ou", type=float, default=DEFAULT_MARKET_THRESHOLDS["OU"], help="EV threshold for over/under bets.")
    parser.add_argument(
        "--monte-carlo-runs",
        type=int,
        default=DEFAULT_MONTE_CARLO_RUNS,
        help="Number of Monte Carlo paths for the PnL simulation.",
    )
    parser.add_argument("--output-dir", default=".", help="Directory where Q2 result files and plots should be written.")
    parser.add_argument(
        "--evaluation-scope",
        default="B",
        choices=["B", "ALL"],
        help="Evaluate on clean partition B only, or on the full betting window (A+B).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG generation.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    cli_market_thresholds = None
    if cli_args.ev_threshold is None:
        cli_market_thresholds = {
            "1X2": cli_args.threshold_1x2,
            "HC": cli_args.threshold_hc,
            "OU": cli_args.threshold_ou,
        }
    run_q2_pipeline(
        data_dir=cli_args.data_dir,
        ev_threshold=cli_args.ev_threshold,
        market_thresholds=cli_market_thresholds,
        monte_carlo_runs=cli_args.monte_carlo_runs,
        output_dir=cli_args.output_dir,
        evaluation_scope=cli_args.evaluation_scope,
        make_plots=not cli_args.no_plots,
    )
