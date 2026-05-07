# Corner Count Bidding 

This repo has two main layers:

- `Q1`: predict home and away corners, plus uncertainty.
- `Q2/Q3/Q4`: turn those predictions into market probabilities, then size bets with Kelly.

## Recommended reading order

If you are new to the repo, read the files in this order:

1. [Latex/ex1_writeup.pdf](Latex/ex1_writeup.pdf)
2. [data_cleaning.ipynb](data_cleaning.ipynb)
3. [q1_pipeline.py](q1_pipeline.py)
4. [team_strength.py](team_strength.py)
5. [team_variance.py](team_variance.py)
6. [market_betting_backtest.py](market_betting_backtest.py)
7. [kelly_backtest.py](kelly_backtest.py)

## How to run

### 1. Build the clean inputs

Run [data_cleaning.ipynb](data_cleaning.ipynb).

It writes:

- [train.parquet](train.parquet)
- [betting.parquet](betting.parquet)
- [all_matches.parquet](all_matches.parquet)

### 2. Run Q1

```bash
python q1_pipeline.py --no-plots
```

If you also want the diagnostic figures:

```bash
python q1_pipeline.py
```

Main Q1 outputs:

- [q1_model_comparison.csv](q1_model_comparison.csv)
- [q1_validation_predictions.csv](q1_validation_predictions.csv)
- [q1_betting_match_predictions.csv](q1_betting_match_predictions.csv)
- [outputs/q1/q1_betting_match_predictions.csv](outputs/q1/q1_betting_match_predictions.csv)

### 3. Run the Kelly backtest

```bash
python kelly_backtest.py --output-dir outputs/kelly_backtest --initial-bankroll 100 --fee-rate 0.03 --min-fee 0.1 --max-bet 5000
```

## Current Kelly assumptions

- `1X2` bet only if `EV > 0.03`
- `HC` bet only if `EV > 0.023`
- `OU` bet only if `EV > 0.03`
- initial bankroll = `100`
- fee rate = `3%`
- minimum fee = `0.1`
- maximum single bet = `5000`
- stop if bankroll hits `0`

## Where to look for results

If you only want the headline outputs, open these:

1. [Latex/ex1_writeup.pdf](Latex/ex1_writeup.pdf)
2. [outputs/kelly_backtest/strategy_summary.csv](outputs/kelly_backtest/strategy_summary.csv)
3. [outputs/kelly_backtest/cumulative_net_pnl.png](outputs/kelly_backtest/cumulative_net_pnl.png)
4. [outputs/kelly_backtest/bankroll_paths_log.png](outputs/kelly_backtest/bankroll_paths_log.png)
5. [outputs/kelly_backtest/selection_edge_monte_carlo.png](outputs/kelly_backtest/selection_edge_monte_carlo.png)
6. [outputs/kelly_backtest/probability_construction.png](outputs/kelly_backtest/probability_construction.png)

## Minimal mental model

1. [data_cleaning.ipynb](data_cleaning.ipynb) prepares the clean match tables.
2. [q1_pipeline.py](q1_pipeline.py) predicts home and away corner means, variances, and quantile anchors.
3. [market_betting_backtest.py](market_betting_backtest.py) converts those Q1 outputs into betting probabilities and selections.
4. [kelly_backtest.py](kelly_backtest.py) decides how hard to press the edge with different Kelly fractions.
