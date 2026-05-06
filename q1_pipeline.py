from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from scipy.stats import nbinom, poisson
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    "p_h_1x2",
    "p_a_1x2",
    "p_d_1x2",
]

CATEGORICAL_COLS = ["competition_id", "season_id"]


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
    print(f"\n=== {title} ===")


def print_frame(name: str, frame: pd.DataFrame, decimals: int = 3) -> None:
    print(f"\n{name}")
    print(frame.round(decimals).to_string())


def irreducible_mae(lam: float, n: int = 200_000) -> float:
    return float(np.mean(np.abs(np.random.poisson(lam, n) - lam)))


def latest_snapshot(
    fixtures: pd.DataFrame,
    side: str,
    snapshot: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    team_col = f"{side}_team_id"
    out = fixtures[["match_id", "date_time", team_col]].rename(columns={team_col: "team_id"}).copy()
    out = out.sort_values("date_time").reset_index(drop=True)
    out = pd.merge_asof(out, snapshot, on="date_time", by="team_id", direction="backward")
    renamed = {col: f"{side}_{col}" for col in cols}
    return out[["match_id"] + cols].rename(columns=renamed)


def poisson_nll(y: pd.Series | np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-6, None)
    return float(-poisson.logpmf(np.asarray(y, int), mu).mean())


def poisson_crps(y: pd.Series | np.ndarray, mu: np.ndarray, k_max: int = 40) -> float:
    y_arr = np.asarray(y, int)
    mu_arr = np.asarray(mu, float)
    ks = np.arange(k_max).reshape(-1, 1)
    cdf = poisson.cdf(ks, mu_arr.reshape(1, -1))
    indicator = (ks >= y_arr.reshape(1, -1)).astype(float)
    return float(np.mean(np.sum((cdf - indicator) ** 2, axis=0)))


def evaluate_predictions(
    name: str,
    y_val_h: pd.Series,
    y_val_a: pd.Series,
    mu_h: np.ndarray,
    mu_a: np.ndarray,
) -> dict[str, float | str]:
    y_val_d = y_val_h - y_val_a
    mu_d = mu_h - mu_a
    return {
        "model": name,
        "home_MAE": mean_absolute_error(y_val_h, mu_h),
        "away_MAE": mean_absolute_error(y_val_a, mu_a),
        "diff_MAE": mean_absolute_error(y_val_d, mu_d),
        "home_RMSE": np.sqrt(mean_squared_error(y_val_h, mu_h)),
        "away_RMSE": np.sqrt(mean_squared_error(y_val_a, mu_a)),
        "diff_RMSE": np.sqrt(mean_squared_error(y_val_d, mu_d)),
        "home_NLL": poisson_nll(y_val_h, mu_h),
        "away_NLL": poisson_nll(y_val_a, mu_a),
        "home_CRPS": poisson_crps(y_val_h, mu_h),
        "away_CRPS": poisson_crps(y_val_a, mu_a),
    }


def load_inputs(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    inv_sum = 1 / x12["oh"] + 1 / x12["oa"] + 1 / x12["od"]
    x12["p_h_1x2"] = (1 / x12["oh"]) / inv_sum
    x12["p_a_1x2"] = (1 / x12["oa"]) / inv_sum
    x12["p_d_1x2"] = (1 / x12["od"]) / inv_sum
    x12 = x12[["match_id", "p_h_1x2", "p_a_1x2", "p_d_1x2"]]

    betting_match_features = (
        betting_matches.merge(bet_h, on="match_id", how="left")
        .merge(bet_a, on="match_id", how="left")
        .merge(strength_features, on="match_id", how="left")
        .merge(x12, on="match_id", how="left")
        .sort_values(["date_time", "competition_id"])
        .reset_index(drop=True)
    )

    for col in ["p_h_1x2", "p_a_1x2", "p_d_1x2"]:
        betting_match_features[col] = betting_match_features[col].fillna(1 / 3)
        features[col] = np.nan

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
    print(
        "NaN after fill -- features (excl. p_*_1x2 on train): "
        f"{features[[col for col in train_feat_cols if not col.endswith('_1x2')]].isna().sum().sum()}"
    )
    print(f"NaN after fill -- betting_match_features: {betting_match_features[bet_feat_cols].isna().sum().sum()}")
    return features, betting_match_features, global_std_c


def split_train_val(
    features: pd.DataFrame,
    betting_match_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    x_train = train_feat[FEATURE_COLS]
    x_val = val_feat[FEATURE_COLS]
    y_train_h = train_feat["home_corners"]
    y_train_a = train_feat["away_corners"]
    y_val_h = val_feat["home_corners"]
    y_val_a = val_feat["away_corners"]
    y_val_d = y_val_h - y_val_a
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

    def fit_single(y_train: pd.Series, y_val: pd.Series) -> LGBMRegressor:
        model = LGBMRegressor(**lgb_kw)
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_val, y_val)],
            callbacks=[early_stopping(150, verbose=False)],
            categorical_feature=CATEGORICAL_COLS,
        )
        return model

    lgb_h = fit_single(y_train_h, y_val_h)
    lgb_a = fit_single(y_train_a, y_val_a)
    mu_h = np.maximum(lgb_h.predict(x_val), 0.1)
    mu_a = np.maximum(lgb_a.predict(x_val), 0.1)
    mu_h_train = np.maximum(lgb_h.predict(x_train), 0.1)
    mu_a_train = np.maximum(lgb_a.predict(x_train), 0.1)

    print_heading("Stage 2")
    print(
        f"lgb best_iter home={lgb_h.best_iteration_} away={lgb_a.best_iteration_}\n"
        f"pred std home={np.std(mu_h):.2f} away={np.std(mu_a):.2f} diff={np.std(mu_h - mu_a):.2f}\n"
        f"pred range home=[{mu_h.min():.2f},{mu_h.max():.2f}] "
        f"away=[{mu_a.min():.2f},{mu_a.max():.2f}] diff=[{(mu_h - mu_a).min():.2f},{(mu_h - mu_a).max():.2f}]\n"
        f"train_MAE home={mean_absolute_error(y_train_h, mu_h_train):.4f} "
        f"away={mean_absolute_error(y_train_a, mu_a_train):.4f}\n"
        f"val_MAE home={mean_absolute_error(y_val_h, mu_h):.4f} "
        f"away={mean_absolute_error(y_val_a, mu_a):.4f} "
        f"diff={mean_absolute_error(y_val_d, mu_h - mu_a):.4f}"
    )

    mean_h = np.full(len(val_feat), y_train_h.mean())
    mean_a = np.full(len(val_feat), y_train_a.mean())
    results = pd.DataFrame(
        [
            evaluate_predictions("Mean baseline", y_val_h, y_val_a, mean_h, mean_a),
            evaluate_predictions("LGB", y_val_h, y_val_a, mu_h, mu_a),
        ]
    ).sort_values("home_MAE").reset_index(drop=True)
    print_frame("validation comparison", results, decimals=4)
    return lgb_h, lgb_a, mu_h, mu_a, mu_h_train, mu_a_train, results


def distance_to_floor(bound: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
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
    lgb_h: LGBMRegressor,
    lgb_a: LGBMRegressor,
    mu_h: np.ndarray,
    mu_a: np.ndarray,
    y_val_h: pd.Series,
    y_val_a: pd.Series,
    global_std_c: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    print_heading("Stage 3")
    for name, mu, y in [("home", mu_h, y_val_h), ("away", mu_a, y_val_a)]:
        lam = float(y.mean())
        ceiling = float(np.sqrt(max(y.var() - lam, 0.01)))
        pred_std = float(np.std(mu))
        print(
            f"{name:6s} actual_std={float(y.std()):.2f} "
            f"ceiling={ceiling:.2f} pred_std={pred_std:.2f} captured={100 * pred_std / ceiling:.0f}%"
        )

    bet_sorted = betting_match_features.sort_values("date_time").reset_index(drop=True)
    n_a = int(len(bet_sorted) * 0.40)
    bet_sorted["partition"] = ["A"] * n_a + ["B"] * (len(bet_sorted) - n_a)
    betting_match_features = bet_sorted

    x_bet = betting_match_features[FEATURE_COLS]
    mu_h_bet = np.maximum(lgb_h.predict(x_bet), 0.1)
    mu_a_bet = np.maximum(lgb_a.predict(x_bet), 0.1)

    a_mask = (betting_match_features["partition"] == "A").values
    y_h_a = betting_match_features.loc[a_mask, "home_corners"].astype(float).values
    y_a_a = betting_match_features.loc[a_mask, "away_corners"].astype(float).values
    mu_h_a = mu_h_bet[a_mask]
    mu_a_a = mu_a_bet[a_mask]
    ph_a = betting_match_features.loc[a_mask, "p_h_1x2"].fillna(1 / 3).values
    pa_a = betting_match_features.loc[a_mask, "p_a_1x2"].fillna(1 / 3).values
    std_h_a = betting_match_features.loc[a_mask, "home_cf_std_l20"].fillna(global_std_c).values
    std_a_a = betting_match_features.loc[a_mask, "away_cf_std_l20"].fillna(global_std_c).values
    market_certainty_h_a = np.abs(ph_a - 0.5)
    market_certainty_a_a = np.abs(pa_a - 0.5)

    dispersion_home = fit_dispersion(mu_h_a, y_h_a, market_certainty_h_a, std_h_a)
    dispersion_away = fit_dispersion(mu_a_a, y_a_a, market_certainty_a_a, std_a_a)

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
            "residual_h": y_h_a - mu_h_a,
            "residual_a": y_a_a - mu_a_a,
        }
    )
    bias = a_resid.groupby("competition_id")[["residual_h", "residual_a"]].mean()
    b_competitions = betting_match_features.loc[~a_mask, "competition_id"]
    correction_h = np.clip(b_competitions.map(bias["residual_h"]).fillna(0).values, -2, 2)
    correction_a = np.clip(b_competitions.map(bias["residual_a"]).fillna(0).values, -2, 2)

    mu_h_bet_cal = mu_h_bet.copy()
    mu_a_bet_cal = mu_a_bet.copy()
    mu_h_bet_cal[~a_mask] += correction_h
    mu_a_bet_cal[~a_mask] += correction_a
    mu_h_bet_cal = np.maximum(mu_h_bet_cal, 0.1)
    mu_a_bet_cal = np.maximum(mu_a_bet_cal, 0.1)

    print("\nPer-competition bias correction (A residuals -> B)")
    print(bias.round(3).to_string())
    print(f"\nB raw:   home={mu_h_bet[~a_mask].mean():.3f} away={mu_a_bet[~a_mask].mean():.3f}")
    print(f"B calib: home={mu_h_bet_cal[~a_mask].mean():.3f} away={mu_a_bet_cal[~a_mask].mean():.3f}")
    mae_h_raw = np.abs(betting_match_features.loc[~a_mask, "home_corners"].values - mu_h_bet[~a_mask]).mean()
    mae_h_cal = np.abs(betting_match_features.loc[~a_mask, "home_corners"].values - mu_h_bet_cal[~a_mask]).mean()
    mae_a_raw = np.abs(betting_match_features.loc[~a_mask, "away_corners"].values - mu_a_bet[~a_mask]).mean()
    mae_a_cal = np.abs(betting_match_features.loc[~a_mask, "away_corners"].values - mu_a_bet_cal[~a_mask]).mean()
    print(f"B MAE home: raw={mae_h_raw:.4f} calib={mae_h_cal:.4f} delta={mae_h_cal - mae_h_raw:+.4f}")
    print(f"B MAE away: raw={mae_a_raw:.4f} calib={mae_a_cal:.4f} delta={mae_a_cal - mae_a_raw:+.4f}")

    ph_all = betting_match_features["p_h_1x2"].fillna(1 / 3).values
    pa_all = betting_match_features["p_a_1x2"].fillna(1 / 3).values
    std_h_all = betting_match_features["home_cf_std_l20"].fillna(global_std_c).values
    std_a_all = betting_match_features["away_cf_std_l20"].fillna(global_std_c).values
    sigma2_h_bet = dispersion_home.predict(mu_h_bet_cal, np.abs(ph_all - 0.5), std_h_all)
    sigma2_a_bet = dispersion_away.predict(mu_a_bet_cal, np.abs(pa_all - 0.5), std_a_all)

    print("\nDispersion calibration audit on partition A (home)")
    sigma2_h_a = dispersion_home.predict(mu_h_a, market_certainty_h_a, std_h_a)
    audit = pd.DataFrame(
        {
            "mu_bucket": pd.qcut(mu_h_a, 5, duplicates="drop"),
            "realised_sq_resid": (y_h_a - mu_h_a) ** 2,
            "pred_sigma2": sigma2_h_a,
        }
    )
    audit_grouped = audit.groupby("mu_bucket", observed=True).agg(
        n=("realised_sq_resid", "size"),
        realised_var=("realised_sq_resid", "mean"),
        pred_var=("pred_sigma2", "mean"),
    )
    audit_grouped["ratio"] = audit_grouped["realised_var"] / audit_grouped["pred_var"]
    print(audit_grouped.round(3).to_string())

    print("\nEdge preservation: (mu_h + mu_a) vs OU line on partition B")
    holdout = betting_match_features[~a_mask].copy()
    mu_total_b = (mu_h_bet_cal + mu_a_bet_cal)[~a_mask]
    ou_b = betting[betting["odds_type"] == "OU"].drop_duplicates("match_id")[["match_id", "od"]]
    ou_b = ou_b.rename(columns={"od": "ou_line"})
    holdout = holdout.merge(ou_b, on="match_id", how="left")
    diff_total = (mu_total_b - holdout["ou_line"]).dropna()
    print(f"B with OU line: n={len(diff_total)} median |mu_t - OU_line|={float(diff_total.abs().median()):.3f}")

    print("\nSpot-check match 12449945")
    row = betting_match_features[betting_match_features["match_id"] == 12449945]
    if len(row):
        pos = list(betting_match_features["match_id"]).index(12449945)
        print(
            f"pred_home={mu_h_bet_cal[pos]:.2f} pred_away={mu_a_bet_cal[pos]:.2f} "
            f"pred_diff={mu_h_bet_cal[pos] - mu_a_bet_cal[pos]:+.2f}"
        )
        print(f"actual: home={int(row.iloc[0].home_corners)} away={int(row.iloc[0].away_corners)}")
        print(f"sigma2_home={sigma2_h_bet[pos]:.2f} sigma2_away={sigma2_a_bet[pos]:.2f}")

    calibration = {
        "dispersion_home": dispersion_home,
        "dispersion_away": dispersion_away,
        "a_mask": a_mask,
        "mu_h_bet_cal": mu_h_bet_cal,
        "mu_a_bet_cal": mu_a_bet_cal,
        "sigma2_h_bet": sigma2_h_bet,
        "sigma2_a_bet": sigma2_a_bet,
        "global_std_c": global_std_c,
        "ph_all": ph_all,
        "pa_all": pa_all,
        "std_h_all": std_h_all,
        "std_a_all": std_a_all,
        "y_h_a": y_h_a,
        "y_a_a": y_a_a,
        "mu_h_a": mu_h_a,
        "mu_a_a": mu_a_a,
    }
    return betting_match_features, mu_h_bet_cal, mu_a_bet_cal, sigma2_h_bet, sigma2_a_bet, calibration


def make_mean_calibration_plot(
    output_dir: Path,
    model_data: pd.DataFrame,
    betting_match_features: pd.DataFrame,
    y_val_h: pd.Series,
    y_val_a: pd.Series,
    mu_h: np.ndarray,
    mu_a: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, y_series, mu, name in [
        (axes[0], y_val_h, mu_h, "Home"),
        (axes[1], y_val_a, mu_a, "Away"),
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
    mu_h: np.ndarray,
    mu_a: np.ndarray,
    calibration: dict[str, object],
) -> dict[str, float]:
    dispersion_home = calibration["dispersion_home"]
    dispersion_away = calibration["dispersion_away"]
    global_std_c = float(calibration["global_std_c"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    inflation_factors: dict[str, float] = {}

    for ax, y_series, mu, std_col, disp_model, name, key in [
        (axes[0], y_val_h, mu_h, "home_cf_std_l20", dispersion_home, "Home", "h"),
        (axes[1], y_val_a, mu_a, "away_cf_std_l20", dispersion_away, "Away", "a"),
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
    a_mask = calibration["a_mask"]
    mu_h_bet_cal = np.asarray(calibration["mu_h_bet_cal"]).copy()
    mu_a_bet_cal = np.asarray(calibration["mu_a_bet_cal"]).copy()
    dispersion_home = calibration["dispersion_home"]
    dispersion_away = calibration["dispersion_away"]
    ph_all = np.asarray(calibration["ph_all"])
    pa_all = np.asarray(calibration["pa_all"])
    std_h_all = np.asarray(calibration["std_h_all"])
    std_a_all = np.asarray(calibration["std_a_all"])

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

    mu_h_bet_cal[a_mask] = np.maximum(mu_h_bet_cal[a_mask] + global_corr_h, 0.1)
    mu_a_bet_cal[a_mask] = np.maximum(mu_a_bet_cal[a_mask] + global_corr_a, 0.1)
    sigma2_h_bet = dispersion_home.predict(mu_h_bet_cal, np.abs(ph_all - 0.5), std_h_all)
    sigma2_a_bet = dispersion_away.predict(mu_a_bet_cal, np.abs(pa_all - 0.5), std_a_all)

    print("\nBetting set: predicted vs actual means after calibration")
    for label, mask in [("A", a_mask), ("B", ~a_mask)]:
        pred_home = mu_h_bet_cal[mask].mean()
        pred_away = mu_a_bet_cal[mask].mean()
        actual_home = betting_match_features.loc[mask, "home_corners"].mean()
        actual_away = betting_match_features.loc[mask, "away_corners"].mean()
        print(
            f"Partition {label}: pred home={pred_home:.3f} (actual {actual_home:.3f}, bias {pred_home - actual_home:+.3f}) "
            f"pred away={pred_away:.3f} (actual {actual_away:.3f}, bias {pred_away - actual_away:+.3f})"
        )
    print(f"sigma2 home: median={np.median(sigma2_h_bet):.2f} away: median={np.median(sigma2_a_bet):.2f}")
    return mu_h_bet_cal, mu_a_bet_cal, sigma2_h_bet, sigma2_a_bet


def save_outputs(
    output_dir: Path,
    val_feat: pd.DataFrame,
    betting_match_features: pd.DataFrame,
    results: pd.DataFrame,
    mu_h: np.ndarray,
    mu_a: np.ndarray,
    mu_h_bet_cal: np.ndarray,
    mu_a_bet_cal: np.ndarray,
    sigma2_h_bet: np.ndarray,
    sigma2_a_bet: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results.to_csv(output_dir / "q1_model_comparison.csv", index=False)

    val_keep = ["match_id", "date_time", "competition_id", "season_id", "home_corners", "away_corners"]
    val_pred = val_feat[val_keep].copy()
    val_pred["pred_home"] = mu_h
    val_pred["pred_away"] = mu_a
    val_pred.to_csv(output_dir / "q1_validation_predictions.csv", index=False)

    bet_pred = betting_match_features[
        ["match_id", "date_time", "competition_id", "season_id", "home_corners", "away_corners", "partition"]
    ].copy()
    bet_pred["pred_home_corners"] = mu_h_bet_cal
    bet_pred["pred_away_corners"] = mu_a_bet_cal
    bet_pred["pred_corner_diff"] = mu_h_bet_cal - mu_a_bet_cal
    bet_pred["sigma2_home"] = sigma2_h_bet
    bet_pred["sigma2_away"] = sigma2_a_bet
    bet_pred.to_csv(output_dir / "q1_betting_match_predictions.csv", index=False)

    print_heading("Save Outputs")
    print("Saved q1_model_comparison.csv, q1_validation_predictions.csv, and q1_betting_match_predictions.csv")
    print(
        f"Betting partition counts: A={(betting_match_features.partition == 'A').sum()} "
        f"B={(betting_match_features.partition == 'B').sum()}\n"
        f"pred_corner_diff: std={(mu_h_bet_cal - mu_a_bet_cal).std():.3f} "
        f"range=[{(mu_h_bet_cal - mu_a_bet_cal).min():.2f},{(mu_h_bet_cal - mu_a_bet_cal).max():.2f}]\n"
        f"sigma2_home: median={np.median(sigma2_h_bet):.2f} sigma2_away: median={np.median(sigma2_a_bet):.2f}"
    )
    return val_pred, bet_pred


def run_pipeline(data_dir: str | Path = ".", save_outputs_flag: bool = True, make_plots: bool = True) -> PipelineArtifacts:
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
    lgb_h, lgb_a, mu_h, mu_a, _, _, results = fit_mean_models(train_feat, val_feat)
    gap = distance_to_floor(bound, results)

    y_val_h = val_feat["home_corners"]
    y_val_a = val_feat["away_corners"]
    betting_match_features, mu_h_bet_cal, mu_a_bet_cal, sigma2_h_bet, sigma2_a_bet, calibration = stage3_calibration(
        betting,
        betting_match_features,
        val_feat,
        model_data,
        lgb_h,
        lgb_a,
        mu_h,
        mu_a,
        y_val_h,
        y_val_a,
        global_std_c,
    )

    if make_plots:
        print_heading("Calibration Plots")
        make_mean_calibration_plot(data_path, model_data, betting_match_features, y_val_h, y_val_a, mu_h, mu_a)
        make_distribution_plot(data_path, val_feat, y_val_h, y_val_a, mu_h, mu_a, calibration)

    mu_h_bet_cal, mu_a_bet_cal, sigma2_h_bet, sigma2_a_bet = apply_global_mean_correction(
        model_data,
        betting_match_features,
        calibration,
    )

    if save_outputs_flag:
        val_predictions, bet_predictions = save_outputs(
            data_path,
            val_feat,
            betting_match_features,
            results,
            mu_h,
            mu_a,
            mu_h_bet_cal,
            mu_a_bet_cal,
            sigma2_h_bet,
            sigma2_a_bet,
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
