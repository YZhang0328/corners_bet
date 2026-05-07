from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import market_betting_backtest as market_backtest


matplotlib.use("Agg")

DEFAULT_INITIAL_BANKROLL = 100.0
DEFAULT_FEE_RATE = 0.03
DEFAULT_MIN_FEE = 0.10
DEFAULT_MAX_BET = 5_000.0
DEFAULT_EVALUATION_SCOPE = "B"
DEFAULT_OUTPUT_DIR = "outputs/kelly_backtest"
KELLY_STRATEGIES = {
    "kelly_100": 1.0,
    "kelly_50": 0.5,
    "kelly_25": 0.25,
}


@dataclass
class KellyBacktestArtifacts:
    """Store the selected bets, path tables, and summary table."""

    selected_bets: pd.DataFrame
    strategy_paths: dict[str, pd.DataFrame]
    strategy_summary: pd.DataFrame


def attach_expected_pnl(
    strategy_path: pd.DataFrame,
    selected_bets: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the per-bet EV so realised and expected PnL can be plotted together."""
    join_keys = ["match_id", "date_time", "odds_type", "side", "odds"]
    expected = selected_bets[join_keys + ["ev_tail"]].copy()
    enriched = strategy_path.merge(expected, on=join_keys, how="left", validate="one_to_one")
    enriched["expected_net_pnl"] = enriched["stake"] * enriched["ev_tail"] - enriched["fee"]
    return enriched


def build_daily_pnl_ev_table(strategy_path_with_ev: pd.DataFrame) -> pd.DataFrame:
    """Collapse one Kelly path into daily realised and expected PnL series."""
    daily = (
        strategy_path_with_ev.assign(day=strategy_path_with_ev["date_time"].dt.floor("D"))
        .groupby("day", as_index=False)
        .agg(
            realized_pnl=("net_pnl", "sum"),
            expected_pnl=("expected_net_pnl", "sum"),
        )
    )
    daily["cumulative_realized_pnl"] = daily["realized_pnl"].cumsum()
    daily["cumulative_expected_pnl"] = daily["expected_pnl"].cumsum()
    daily = daily.set_index("day")
    daily["rolling_14d_realized_avg"] = daily["realized_pnl"].rolling("14D").mean()
    daily["rolling_14d_expected_avg"] = daily["expected_pnl"].rolling("14D").mean()
    return daily.reset_index()


def simulate_unit_stake_total_pnl(
    win_probabilities: np.ndarray,
    decimal_odds: np.ndarray,
    runs: int,
    seed: int,
) -> np.ndarray:
    """Simulate total PnL under flat unit stakes for one set of win probabilities."""
    rng = np.random.default_rng(seed)
    wins = rng.binomial(1, np.clip(win_probabilities, 1e-6, 1 - 1e-6), size=(runs, len(win_probabilities)))
    win_pnl = decimal_odds - 1.0
    lose_pnl = -1.0
    return np.where(wins == 1, win_pnl, lose_pnl).sum(axis=1)


def choose_stress_shrink(
    selected_bets: pd.DataFrame,
    base_samples: np.ndarray,
    target_percentile: float = 5.0,
) -> float:
    """Choose how hard to shrink model probabilities toward market probabilities for the stress test."""
    base_probabilities = selected_bets["p_tail"].to_numpy(dtype=float)
    market_probabilities = selected_bets["p_market"].to_numpy(dtype=float)
    decimal_odds = selected_bets["odds"].to_numpy(dtype=float)
    grid = np.linspace(0.40, 1.00, 61)
    best_alpha = 1.0
    best_gap = float("inf")
    for alpha in grid:
        stress_probabilities = np.clip(
            market_probabilities + alpha * (base_probabilities - market_probabilities),
            1e-6,
            1 - 1e-6,
        )
        expected_total_pnl = float(np.sum(stress_probabilities * (decimal_odds - 1.0) - (1.0 - stress_probabilities)))
        percentile = 100.0 * float(np.mean(base_samples <= expected_total_pnl))
        gap = abs(percentile - target_percentile)
        if gap < best_gap:
            best_gap = gap
            best_alpha = float(alpha)
    return best_alpha


def build_selection_edge_stress_summary(
    selected_bets: pd.DataFrame,
    runs: int = 10_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the base and stressed Monte Carlo summaries for the selected flat-stake edge."""
    base_probabilities = selected_bets["p_tail"].to_numpy(dtype=float)
    market_probabilities = selected_bets["p_market"].to_numpy(dtype=float)
    decimal_odds = selected_bets["odds"].to_numpy(dtype=float)
    actual_total_pnl = float(np.where(selected_bets["won"].to_numpy(dtype=float) == 1.0, decimal_odds - 1.0, -1.0).sum())

    base_samples = simulate_unit_stake_total_pnl(base_probabilities, decimal_odds, runs, seed=7)
    stress_alpha = choose_stress_shrink(selected_bets, base_samples, target_percentile=5.0)
    stress_probabilities = np.clip(
        market_probabilities + stress_alpha * (base_probabilities - market_probabilities),
        1e-6,
        1 - 1e-6,
    )
    stress_samples = simulate_unit_stake_total_pnl(stress_probabilities, decimal_odds, runs, seed=11)

    summary = pd.DataFrame(
        [
            {
                "actual_total_pnl": actual_total_pnl,
                "base_median_pnl": float(np.median(base_samples)),
                "base_q05_pnl": float(np.quantile(base_samples, 0.05)),
                "base_q95_pnl": float(np.quantile(base_samples, 0.95)),
                "stress_definition": f"p_stress = p_market + {stress_alpha:.2f} * (p_tail - p_market)",
                "stress_median_pnl": float(np.median(stress_samples)),
                "stress_q05_pnl": float(np.quantile(stress_samples, 0.05)),
                "stress_q95_pnl": float(np.quantile(stress_samples, 0.95)),
                "stress_median_percentile_in_base_mc": round(100.0 * float(np.mean(base_samples <= np.median(stress_samples))), 2),
            }
        ]
    )
    plot_payload = {
        "base_samples": base_samples,
        "stress_samples": stress_samples,
        "actual_total_pnl": actual_total_pnl,
        "base_median_pnl": float(np.median(base_samples)),
        "stress_median_pnl": float(np.median(stress_samples)),
        "stress_definition": summary.iloc[0]["stress_definition"],
    }
    return summary, plot_payload


def print_heading(title: str) -> None:
    """Print a small console section header."""
    print(f"\n=== {title} ===")


def load_selected_bets(
    data_dir: Path,
    evaluation_scope: str,
    market_thresholds: dict[str, float],
) -> pd.DataFrame:
    """Rebuild the final selected-bet table from the current Q1 outputs."""
    predicted_matches, market_prices, observed_results = market_backtest.load_prediction_and_market_tables(data_dir)
    predicted_matches = market_backtest.attach_q1_distribution_columns(predicted_matches)
    partition_a_predictions = predicted_matches[predicted_matches["partition"] == "A"].copy()
    partition_b_predictions = predicted_matches[predicted_matches["partition"] == "B"].copy()

    shared_rho_info = market_backtest.estimate_home_away_corner_correlation(partition_a_predictions)
    partition_a_raw = market_backtest.attach_observed_outcomes(
        market_backtest.build_candidate_bet_rows(partition_a_predictions, market_prices, shared_rho_info["rho"], "A"),
        observed_results,
    )
    partition_b_raw = market_backtest.attach_observed_outcomes(
        market_backtest.build_candidate_bet_rows(partition_b_predictions, market_prices, shared_rho_info["rho"], "B"),
        observed_results,
    )

    base_calibrators, _ = market_backtest.fit_side_probability_calibrators(partition_a_raw)
    partition_a_calibrated = market_backtest.attach_market_no_vig_probabilities(
        market_backtest.apply_side_probability_calibration(partition_a_raw, base_calibrators)
    )
    partition_b_calibrated = market_backtest.attach_market_no_vig_probabilities(
        market_backtest.apply_side_probability_calibration(partition_b_raw, base_calibrators)
    )

    tail_lambda_map = {
        market_name: market_backtest.fit_tail_lambda(partition_a_calibrated, market_name, market_thresholds[market_name])
        for market_name in market_backtest.TAIL_SHRINK_MARKETS
    }
    tail_scale, _ = market_backtest.choose_tail_scale(partition_a_calibrated, tail_lambda_map, market_thresholds)
    partition_a_final = market_backtest.attach_market_thresholds(
        market_backtest.apply_tail_probability_shrink(partition_a_calibrated, tail_lambda_map, tail_scale),
        market_thresholds,
    )
    partition_b_final = market_backtest.attach_market_thresholds(
        market_backtest.apply_tail_probability_shrink(partition_b_calibrated, tail_lambda_map, tail_scale),
        market_thresholds,
    )

    if evaluation_scope.upper() == "ALL":
        evaluation_pool = pd.concat([partition_a_final, partition_b_final], ignore_index=True)
    else:
        evaluation_pool = partition_b_final.copy()

    selected_bets = market_backtest.select_bets_with_market_thresholds(evaluation_pool, "ev_tail")
    selected_bets = selected_bets.sort_values("date_time").reset_index(drop=True)

    print_heading("Selected Bets")
    print(f"scope={evaluation_scope.upper()} | bets={len(selected_bets)} | breakdown={selected_bets['odds_type'].value_counts().to_dict()}")
    return selected_bets


def full_kelly_fraction(win_probability: np.ndarray, decimal_odds: np.ndarray) -> np.ndarray:
    """Convert win probabilities into full Kelly fractions."""
    net_odds = np.maximum(decimal_odds - 1.0, 1e-12)
    raw_fraction = (win_probability * decimal_odds - 1.0) / net_odds
    return np.clip(raw_fraction, 0.0, 1.0)


def allocate_group_stakes(
    bankroll: float,
    full_kelly: np.ndarray,
    fraction_scale: float,
    fee_rate: float,
    min_fee: float,
    max_bet: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn Kelly fractions into stake sizes after fees and caps."""
    target_fraction = np.clip(full_kelly * fraction_scale, 0.0, 1.0)
    raw_stakes = bankroll * target_fraction
    stakes = np.minimum(raw_stakes, max_bet)
    fees = np.maximum(stakes * fee_rate, min_fee)

    for _ in range(3):
        total_cost = float(np.sum(stakes + fees))
        if total_cost <= bankroll or total_cost <= 0:
            break
        stakes *= bankroll / total_cost
        fees = np.maximum(stakes * fee_rate, min_fee)

    return target_fraction, raw_stakes, stakes, fees


def simulate_kelly_path(
    selected_bets: pd.DataFrame,
    strategy_name: str,
    fraction_scale: float,
    initial_bankroll: float,
    fee_rate: float,
    min_fee: float,
    max_bet: float,
) -> tuple[pd.DataFrame, dict[str, float | int | str | bool]]:
    """Replay one realised Kelly bankroll path and stop as soon as bankroll hits zero."""
    bets = selected_bets.sort_values("date_time").reset_index(drop=True).copy()
    bankroll = float(initial_bankroll)
    peak_bankroll = bankroll
    busted = False
    bust_date = ""
    records: list[dict[str, object]] = []

    full_fraction = full_kelly_fraction(bets["p_tail"].to_numpy(dtype=float), bets["odds"].to_numpy(dtype=float))

    for date_time, group in bets.groupby("date_time", sort=True):
        if bankroll <= 1e-12:
            busted = True
            bust_date = pd.Timestamp(date_time).isoformat()
            break

        group_indices = group.index.to_numpy(dtype=int)
        bankroll_before_group = bankroll
        target_fraction, raw_stakes, stakes, fees = allocate_group_stakes(
            bankroll=bankroll_before_group,
            full_kelly=full_fraction[group_indices],
            fraction_scale=fraction_scale,
            fee_rate=fee_rate,
            min_fee=min_fee,
            max_bet=max_bet,
        )

        odds = group["odds"].to_numpy(dtype=float)
        wins = group["won"].to_numpy(dtype=float)
        net_pnl = np.where(wins == 1.0, stakes * (odds - 1.0) - fees, -stakes - fees)
        cumulative_group_pnl = np.cumsum(net_pnl)
        bankroll = max(bankroll_before_group + float(net_pnl.sum()), 0.0)

        for pos, group_index in enumerate(group_indices):
            row = bets.loc[group_index]
            bankroll_after = max(bankroll_before_group + float(cumulative_group_pnl[pos]), 0.0)
            peak_bankroll = max(peak_bankroll, bankroll_after)
            drawdown = (bankroll_after - peak_bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0
            records.append(
                {
                    "strategy": strategy_name,
                    "bet_no": len(records) + 1,
                    "date_time": row["date_time"],
                    "match_id": row["match_id"],
                    "odds_type": row["odds_type"],
                    "side": row["side"],
                    "odds": row["odds"],
                    "won": row["won"],
                    "bankroll_before": bankroll_before_group,
                    "full_kelly_fraction": float(full_fraction[group_index]),
                    "bet_fraction": float(target_fraction[pos]),
                    "raw_stake_before_cap": float(raw_stakes[pos]),
                    "stake": float(stakes[pos]),
                    "fee": float(fees[pos]),
                    "net_pnl": float(net_pnl[pos]),
                    "bankroll_after": bankroll_after,
                    "drawdown": float(drawdown),
                    "hit_bet_cap": bool(raw_stakes[pos] > max_bet),
                }
            )

        if bankroll <= 1e-12:
            busted = True
            bust_date = pd.Timestamp(date_time).isoformat()
            bankroll = 0.0
            break

    path = pd.DataFrame(records)
    total_staked = float(path["stake"].sum()) if len(path) else 0.0
    total_fees = float(path["fee"].sum()) if len(path) else 0.0
    total_pnl = float(path["net_pnl"].sum()) if len(path) else 0.0
    ending_bankroll = float(path["bankroll_after"].iloc[-1]) if len(path) else float(initial_bankroll)
    summary = {
        "strategy": strategy_name,
        "initial_bankroll": float(initial_bankroll),
        "fee_rate": float(fee_rate),
        "min_fee": float(min_fee),
        "max_bet": float(max_bet),
        "available_bets": int(len(selected_bets)),
        "bets_placed": int(len(path)),
        "bets_skipped_after_bust": int(len(selected_bets) - len(path)),
        "bets_hitting_cap": int(path["hit_bet_cap"].sum()) if len(path) else 0,
        "total_staked": total_staked,
        "total_fees": total_fees,
        "total_pnl": total_pnl,
        "ending_bankroll": ending_bankroll,
        "bankroll_return": ending_bankroll / float(initial_bankroll) - 1.0,
        "turnover_roi": total_pnl / total_staked if total_staked else np.nan,
        "max_drawdown": float(path["drawdown"].min()) if len(path) else np.nan,
        "busted": bool(busted),
        "bust_date": bust_date,
        "max_stake": float(path["stake"].max()) if len(path) else 0.0,
        "avg_stake": float(path["stake"].mean()) if len(path) else 0.0,
    }
    return path, summary


def daily_return_sharpe_proxy(strategy_path: pd.DataFrame) -> float:
    """Estimate a simple Sharpe-like ratio from daily bankroll returns."""
    if strategy_path.empty:
        return np.nan
    daily = (
        strategy_path.groupby(strategy_path["date_time"].dt.date)
        .agg(day_start_bankroll=("bankroll_before", "first"), day_pnl=("net_pnl", "sum"))
        .reset_index(drop=True)
    )
    if len(daily) < 2:
        return np.nan
    daily_returns = daily["day_pnl"] / daily["day_start_bankroll"].replace(0.0, np.nan)
    daily_std = float(daily_returns.std(ddof=1))
    if np.isnan(daily_std) or daily_std <= 0.0:
        return np.nan
    return float(np.sqrt(252.0) * daily_returns.mean() / daily_std)


def plot_bankroll_paths(strategy_paths: dict[str, pd.DataFrame], output_dir: Path, initial_bankroll: float) -> None:
    """Save the Kelly bankroll charts."""
    colors = {"kelly_100": "#c0392b", "kelly_50": "#e67e22", "kelly_25": "#2980b9"}
    labels = {"kelly_100": "100% Kelly", "kelly_50": "50% Kelly", "kelly_25": "25% Kelly"}

    fig, axis = plt.subplots(figsize=(12, 5))
    for name, frame in strategy_paths.items():
        axis.plot(frame["date_time"], frame["bankroll_after"], color=colors[name], linewidth=1.7, label=labels[name])
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axis.set_title("Kelly bankroll path (stops at zero)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Bankroll")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.legend()
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "bankroll_paths_linear.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    for name, frame in strategy_paths.items():
        axis.plot(frame["date_time"], frame["bankroll_after"].clip(lower=1e-6), color=colors[name], linewidth=1.7, label=labels[name])
    axis.set_yscale("log")
    axis.set_title("Kelly bankroll path (log scale)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Bankroll (log)")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.legend()
    axis.grid(alpha=0.25, which="both")
    plt.tight_layout()
    plt.savefig(output_dir / "bankroll_paths_log.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    for name, frame in strategy_paths.items():
        axis.plot(frame["date_time"], frame["bankroll_after"] - initial_bankroll, color=colors[name], linewidth=1.7, label=labels[name])
    axis.axhline(-initial_bankroll, color="black", linewidth=1.0, alpha=0.5)
    axis.set_title("Kelly cumulative net PnL")
    axis.set_xlabel("Date")
    axis.set_ylabel("Net PnL")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.legend()
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_net_pnl.png", dpi=160)
    plt.close(fig)


def plot_q3_pnl_ev_views(kelly_25_daily: pd.DataFrame, output_dir: Path) -> None:
    """Save the cumulative and 14-day rolling PnL/EV plots for the 25% Kelly path."""
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        kelly_25_daily["day"],
        kelly_25_daily["cumulative_realized_pnl"],
        linewidth=2.0,
        label="Cumulative realized PnL",
    )
    axis.plot(
        kelly_25_daily["day"],
        kelly_25_daily["cumulative_expected_pnl"],
        linewidth=2.0,
        label="Cumulative expected PnL (EV)",
    )
    axis.set_title("25% Kelly cumulative PnL and EV by day")
    axis.set_xlabel("Date")
    axis.set_ylabel("USD")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.legend()
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "kelly_25_cumulative_pnl_ev.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        kelly_25_daily["day"],
        kelly_25_daily["rolling_14d_realized_avg"],
        linewidth=2.0,
        label="14-day rolling realized PnL average",
    )
    axis.plot(
        kelly_25_daily["day"],
        kelly_25_daily["rolling_14d_expected_avg"],
        linewidth=2.0,
        label="14-day rolling EV average",
    )
    axis.set_title("25% Kelly 14-day rolling PnL and EV averages")
    axis.set_xlabel("Date")
    axis.set_ylabel("USD per day")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.legend()
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "kelly_25_rolling_pnl_ev.png", dpi=160)
    plt.close(fig)


def plot_selection_edge_monte_carlo(plot_payload: dict[str, object], output_dir: Path) -> None:
    """Plot the base Monte Carlo edge distribution and one stressed scenario."""
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.hist(
        plot_payload["base_samples"],
        bins=80,
        density=True,
        alpha=0.70,
        color="steelblue",
        label="Base MC distribution",
    )
    axis.axvline(
        plot_payload["actual_total_pnl"],
        color="red",
        linewidth=2.0,
        label=f"Actual PnL = {plot_payload['actual_total_pnl']:+.2f}",
    )
    axis.axvline(
        plot_payload["base_median_pnl"],
        color="orange",
        linewidth=1.5,
        linestyle="--",
        label=f"Base MC median = {plot_payload['base_median_pnl']:+.2f}",
    )
    axis.axvline(
        plot_payload["stress_median_pnl"],
        color="purple",
        linewidth=1.8,
        linestyle="-.",
        label=f"Stress median = {plot_payload['stress_median_pnl']:+.2f}",
    )
    axis.axvline(0.0, color="grey", linewidth=0.8, linestyle=":")
    axis.set_title("Monte Carlo total PnL with one stressed-edge scenario")
    axis.set_xlabel("Total PnL (flat unit stakes)")
    axis.set_ylabel("Density")
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "selection_edge_monte_carlo.png", dpi=160)
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    selected_bets: pd.DataFrame,
    strategy_paths: dict[str, pd.DataFrame],
    strategy_summary: pd.DataFrame,
    kelly_25_daily: pd.DataFrame,
    stress_summary: pd.DataFrame,
) -> None:
    """Write the Kelly backtest outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_bets.to_csv(output_dir / "selected_bets.csv", index=False)
    for strategy_name, path in strategy_paths.items():
        path.to_csv(output_dir / f"{strategy_name}_path.csv", index=False)
    strategy_summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    kelly_25_daily.to_csv(output_dir / "kelly_25_daily_pnl_ev.csv", index=False)
    stress_summary.to_csv(output_dir / "selection_edge_stress_summary.csv", index=False)


def run_kelly_backtest(
    data_dir: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    evaluation_scope: str = DEFAULT_EVALUATION_SCOPE,
    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL,
    fee_rate: float = DEFAULT_FEE_RATE,
    min_fee: float = DEFAULT_MIN_FEE,
    max_bet: float = DEFAULT_MAX_BET,
    make_plots: bool = True,
) -> KellyBacktestArtifacts:
    """Run the Kelly bankroll backtest on the current selected bets."""
    data_path = Path(data_dir)
    result_path = Path(output_dir)
    market_thresholds = market_backtest.DEFAULT_MARKET_THRESHOLDS.copy()
    selected_bets = load_selected_bets(data_path, evaluation_scope, market_thresholds)

    strategy_paths: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | int | str | bool]] = []
    for strategy_name, fraction_scale in KELLY_STRATEGIES.items():
        path, summary = simulate_kelly_path(
            selected_bets=selected_bets,
            strategy_name=strategy_name,
            fraction_scale=fraction_scale,
            initial_bankroll=initial_bankroll,
            fee_rate=fee_rate,
            min_fee=min_fee,
            max_bet=max_bet,
        )
        summary["daily_return_sharpe_proxy"] = daily_return_sharpe_proxy(path)
        strategy_paths[strategy_name] = path
        summary_rows.append(summary)

    strategy_summary = pd.DataFrame(summary_rows)
    print_heading("Kelly Summary")
    print(strategy_summary.round(4).to_string(index=False))

    kelly_25_with_ev = attach_expected_pnl(strategy_paths["kelly_25"], selected_bets)
    kelly_25_daily = build_daily_pnl_ev_table(kelly_25_with_ev)
    stress_summary, stress_plot_payload = build_selection_edge_stress_summary(selected_bets)

    save_outputs(result_path, selected_bets, strategy_paths, strategy_summary, kelly_25_daily, stress_summary)
    if make_plots:
        plot_bankroll_paths(strategy_paths, result_path, initial_bankroll)
        plot_q3_pnl_ev_views(kelly_25_daily, result_path)
        plot_selection_edge_monte_carlo(stress_plot_payload, result_path)

    return KellyBacktestArtifacts(
        selected_bets=selected_bets,
        strategy_paths=strategy_paths,
        strategy_summary=strategy_summary,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a Kelly-only bankroll backtest on the current market-backtest selections.")
    parser.add_argument("--data-dir", default=".", help="Directory containing the current Q1 outputs and market files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where Kelly outputs will be written.")
    parser.add_argument("--evaluation-scope", choices=["B", "ALL"], default=DEFAULT_EVALUATION_SCOPE, help="Use clean B only or the full A+B betting window.")
    parser.add_argument("--initial-bankroll", type=float, default=DEFAULT_INITIAL_BANKROLL, help="Starting bankroll.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Proportional fee rate charged per bet.")
    parser.add_argument("--min-fee", type=float, default=DEFAULT_MIN_FEE, help="Minimum fee charged per bet.")
    parser.add_argument("--max-bet", type=float, default=DEFAULT_MAX_BET, help="Maximum stake allowed on a single bet.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_kelly_backtest(
        data_dir=cli_args.data_dir,
        output_dir=cli_args.output_dir,
        evaluation_scope=cli_args.evaluation_scope,
        initial_bankroll=cli_args.initial_bankroll,
        fee_rate=cli_args.fee_rate,
        min_fee=cli_args.min_fee,
        max_bet=cli_args.max_bet,
        make_plots=not cli_args.no_plots,
    )
