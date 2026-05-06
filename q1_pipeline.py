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


TEAM_FEATURE_COLS = [
    "cf_ewm_hl5",
    "ca_ewm_hl5",
    "gf_l20",
    "ga_l20",
    "gd_l20",
    "cf_std_l20",
    "ca_std_l20",
]

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

FEATURE_COLS = [
    "competition_id",
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
    "league_log_baseline_home",
    "league_log_baseline_away",
    "prior_match_count_home",
    "prior_match_count_away",
    "home_cf_ewm_hl5",
    "home_ca_ewm_hl5",
    "away_cf_ewm_hl5",
    "away_ca_ewm_hl5",
    "home_gf_l20",
    "home_ga_l20",
    "home_gd_l20",
    "away_gf_l20",
    "away_ga_l20",
    "away_gd_l20",
    "home_cf_std_l20",
    "home_ca_std_l20",
    "away_cf_std_l20",
    "away_ca_std_l20",
]

CATEGORICAL_COLS = ["competition_id", "season_id"]
MARKET_TWO_WAY_FILL = 1 / 3
MARKET_TARGET_MIN_TOTAL = 0.1
MARKET_TARGET_MAX_TOTAL = 30.0


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
    """Print a section header for console logs."""
    print(f"\n=== {title} ===")


def print_frame(name: str, frame: pd.DataFrame, decimals: int = 3) -> None:
    """Print a rounded dataframe with a preceding label."""
    print(f"\n{name}")
    print(frame.round(decimals).to_string())


def write_csv_with_fallback(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV, falling back to `*.latest.csv` if the main file is locked."""
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.latest{path.suffix}")
        frame.to_csv(fallback, index=False)
        print(f"Permission denied for {path.name}; wrote {fallback.name} instead.")
        return fallback


def irreducible_mae(lam: float, n: int = 200_000) -> float:
    """Approximate the Poisson irreducible MAE floor by Monte Carlo sampling."""
    return float(np.mean(np.abs(np.random.poisson(lam, n) - lam)))


def latest_snapshot(
    fixtures: pd.DataFrame,
    side: str,
    snapshot: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Attach the latest per-team snapshot before each fixture for one side."""
    team_col = f"{side}_team_id"
    out = fixtures[["match_id", "date_time", team_col]].rename(columns={team_col: "team_id"}).copy()
    out = out.sort_values("date_time").reset_index(drop=True)
    out = pd.merge_asof(out, snapshot, on="date_time", by="team_id", direction="backward")
    renamed = {col: f"{side}_{col}" for col in cols}
    return out[["match_id"] + cols].rename(columns=renamed)


def poisson_nll(y: pd.Series | np.ndarray, mu: np.ndarray) -> float:
    """Compute average Poisson negative log-likelihood."""
    mu = np.clip(mu, 1e-6, None)
    return float(-poisson.logpmf(np.asarray(y, int), mu).mean())


def poisson_crps(y: pd.Series | np.ndarray, mu: np.ndarray, k_max: int = 40) -> float:
    """Approximate Poisson CRPS by summing squared CDF errors on a finite grid."""
    y_arr = np.asarray(y, int)
    mu_arr = np.asarray(mu, float)
    ks = np.arange(k_max).reshape(-1, 1)
    cdf = poisson.cdf(ks, mu_arr.reshape(1, -1))
    indicator = (ks >= y_arr.reshape(1, -1)).astype(float)
    return float(np.mean(np.sum((cdf - indicator) ** 2, axis=0)))


def devig_two_way(price_over: float, price_under: float) -> tuple[float, float] | None:
    """Convert 2-way decimal prices into no-vig implied probabilities."""
    if pd.isna(price_over) or pd.isna(price_under) or price_over <= 1 or price_under <= 1:
        return None
    inv_over = 1 / price_over
    inv_under = 1 / price_under
    total = inv_over + inv_under
    if total <= 0:
        return None
    return inv_over / total, inv_under / total


def devig_three_way(price_home: float, price_away: float, price_draw: float) -> tuple[float, float, float] | None:
    """Convert 3-way decimal prices into no-vig implied probabilities."""
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
    """Invert a half-line OU market into an implied total-corners mean."""
    if pd.isna(line) or pd.isna(prob_over):
        return None
    fractional = round(float(line) % 1, 3)
    if fractional != 0.5:
        return None
    threshold = int(np.floor(line))

    def objective(total_mean: float) -> float:
        return float(1 - poisson.cdf(threshold, total_mean) - prob_over)

    try:
        return float(brentq(objective, MARKET_TARGET_MIN_TOTAL, MARKET_TARGET_MAX_TOTAL))
    except ValueError:
        return None


def skellam_probabilities(mu_home: float, mu_away: float) -> tuple[float, float, float]:
    """Return home-win, away-win, and draw probabilities for a Skellam model."""
    prob_draw = float(skellam.pmf(0, mu_home, mu_away))
    prob_away = float(skellam.cdf(-1, mu_home, mu_away))
    prob_home = float(1 - prob_draw - prob_away)
    return prob_home, prob_away, prob_draw


def solve_home_away_means_from_market(
    total_mean: float,
    prob_home: float,
    prob_away: float,
    prob_draw: float,
) -> tuple[float, float] | None:
    """Split a total-corners mean into home/away means that match 1X2 prices."""
    if total_mean <= 0:
        return None

    def loss(mu_home: float) -> float:
        mu_away = total_mean - mu_home
        if mu_home <= 0 or mu_away <= 0:
            return 1e9
        model_home, model_away, model_draw = skellam_probabilities(mu_home, mu_away)
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
    mu_home = float(fit.x)
    mu_away = float(total_mean - mu_home)
    if mu_home <= 0 or mu_away <= 0:
        return None
    return mu_home, mu_away


def infer_market_means_for_row(row: pd.Series) -> tuple[float, float] | None:
    """Infer market-implied home/away corner means from OU and 1X2 prices."""
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


def fit_market_teacher(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> LGBMRegressor:
    """Fit a LightGBM teacher that mimics market-implied corner means."""
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
    """Choose a simple blend weight that minimises MAE on a small grid."""
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
    """Compute validation metrics for home, away, and corner-difference targets."""
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
    """Load cleaned parquet inputs and align the training universe to betting leagues."""
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
    """Summarise raw dispersion and the irreducible Poisson MAE floor."""
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
    """Generate leak-free team-strength features for train and betting fixtures."""
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
    """Convert match rows into one row per team appearance."""
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

    print_heading("Team-Game Long Format")
    print(
        f"match-level rows: {len(model_data):,} -> team-game rows: {len(team_games):,} (= 2 x match rows)"
    )
    print(team_games.head(4).to_string(index=False))
    return team_games


def add_rolling_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add rolling attacking, defensive, and goal-based team features."""
    team_games = team_games.copy()
    team_games["cf_ewm_hl5"] = team_games.groupby("team_id")["corners_for"].transform(
        lambda series: series.shift(1).ewm(halflife=5, ignore_na=True).mean()
    )
    team_games["ca_ewm_hl5"] = team_games.groupby("team_id")["corners_against"].transform(
        lambda series: series.shift(1).ewm(halflife=5, ignore_na=True).mean()
    )
    team_games["gf_l20"] = team_games.groupby("team_id")["goals_for"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=3).mean()
    )
    team_games["ga_l20"] = team_games.groupby("team_id")["goals_against"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=3).mean()
    )
    team_games["gd_l20"] = team_games["gf_l20"] - team_games["ga_l20"]
    team_games["cf_std_l20"] = team_games.groupby("team_id")["corners_for"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=5).std()
    )
    team_games["ca_std_l20"] = team_games.groupby("team_id")["corners_against"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=5).std()
    )
    team_games["n_prior"] = team_games.groupby("team_id").cumcount()

    nan_summary = team_games[
        ["cf_ewm_hl5", "ca_ewm_hl5", "gf_l20", "ga_l20", "cf_std_l20", "ca_std_l20"]
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
    """Assemble the final match-level design matrices for train and betting sets."""
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

    betting_match_features = (
        betting_matches.merge(bet_h, on="match_id", how="left")
        .merge(bet_a, on="match_id", how="left")
        .merge(strength_features, on="match_id", how="left")
        .merge(x12, on="match_id", how="left")
        .merge(ou, on="match_id", how="left")
        .sort_values(["date_time", "competition_id"])
        .reset_index(drop=True)
    )

    for col in ["p_h_1x2", "p_a_1x2", "p_d_1x2"]:
        betting_match_features[col] = betting_match_features[col].fillna(1 / 3)

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
    """Fill engineered-feature nulls and report the remaining data quality picture."""
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
        frame["home_cf_ewm_hl5"] = frame["home_cf_ewm_hl5"].fillna(global_mean_h)
        frame["home_ca_ewm_hl5"] = frame["home_ca_ewm_hl5"].fillna(global_mean_a)
        frame["away_cf_ewm_hl5"] = frame["away_cf_ewm_hl5"].fillna(global_mean_a)
        frame["away_ca_ewm_hl5"] = frame["away_ca_ewm_hl5"].fillna(global_mean_h)

        for col in ["home_gf_l20", "home_ga_l20", "away_gf_l20", "away_ga_l20"]:
            frame[col] = frame[col].fillna(global_mean_g)

        frame["home_gd_l20"] = frame["home_gf_l20"] - frame["home_ga_l20"]
        frame["away_gd_l20"] = frame["away_gf_l20"] - frame["away_ga_l20"]

        for col in ["home_cf_std_l20", "home_ca_std_l20", "away_cf_std_l20", "away_ca_std_l20"]:
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
    """Split the match feature table into chronological train and validation sets."""
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
    """Fit the stage-2 home and away corner mean models."""
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
    """Compare the model's MAE against the baseline and Poisson error floor."""
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
    """Fit dispersion heads and bias-correct holdout mean predictions."""
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
    market_home_prob_a = betting_match_features.loc[a_mask, "p_h_1x2"].fillna(1 / 3).values
    market_away_prob_a = betting_match_features.loc[a_mask, "p_a_1x2"].fillna(1 / 3).values
    rolling_home_std_a = betting_match_features.loc[a_mask, "home_cf_std_l20"].fillna(global_std_c).values
    rolling_away_std_a = betting_match_features.loc[a_mask, "away_cf_std_l20"].fillna(global_std_c).values
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

    market_home_prob_all = betting_match_features["p_h_1x2"].fillna(1 / 3).values
    market_away_prob_all = betting_match_features["p_a_1x2"].fillna(1 / 3).values
    rolling_home_std_all = betting_match_features["home_cf_std_l20"].fillna(global_std_c).values
    rolling_away_std_all = betting_match_features["away_cf_std_l20"].fillna(global_std_c).values
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
    """Create mean-calibration and trend charts for the Q1 mean model."""
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
    """Visualise empirical vs modeled count distributions on validation data."""
    dispersion_home = calibration["dispersion_home"]
    dispersion_away = calibration["dispersion_away"]
    global_std_c = float(calibration["global_std_c"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    inflation_factors: dict[str, float] = {}

    for ax, y_series, mu, std_col, disp_model, name, key in [
        (axes[0], y_val_h, predicted_home_val, "home_cf_std_l20", dispersion_home, "Home", "h"),
        (axes[1], y_val_a, predicted_away_val, "away_cf_std_l20", dispersion_away, "Away", "a"),
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
    """Add a late-window drift correction to partition-A betting predictions."""
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
    """Blend base Q1 predictions with a market-implied teacher on partition B."""
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
    """Persist Q1 validation and betting predictions to CSV files."""
    model_path = write_csv_with_fallback(results, output_dir / "q1_model_comparison.csv")

    val_keep = ["match_id", "date_time", "competition_id", "season_id", "home_corners", "away_corners"]
    val_pred = val_feat[val_keep].copy()
    val_pred["pred_home"] = predicted_home_val
    val_pred["pred_away"] = predicted_away_val
    val_path = write_csv_with_fallback(val_pred, output_dir / "q1_validation_predictions.csv")

    bet_pred = betting_match_features[
        ["match_id", "date_time", "competition_id", "season_id", "home_corners", "away_corners", "partition"]
    ].copy()
    bet_pred["pred_home_corners"] = predicted_home_bet
    bet_pred["pred_away_corners"] = predicted_away_bet
    bet_pred["pred_corner_diff"] = predicted_home_bet - predicted_away_bet
    bet_pred["sigma2_home"] = predicted_home_variance
    bet_pred["sigma2_away"] = predicted_away_variance
    bet_path = write_csv_with_fallback(bet_pred, output_dir / "q1_betting_match_predictions.csv")

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
    """Run the full Q1 feature, mean, calibration, and export pipeline."""
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
