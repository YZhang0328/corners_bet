# Ex1 Corner Modelling and Kelly Backtest

This repo answers `Ex1` in two layers:

- `Q1`: predict home and away corners, plus uncertainty.
- `Q2/Q3/Q4`: turn those predictions into market probabilities, filter bets by EV, and size bets with Kelly.

## Recommended reading order

If you are new to the repo, read the files in this order:

1. [Latex/ex1_writeup.pdf](Latex/ex1_writeup.pdf)  
   The final writeup. Read this first if you want the overall story.

2. [Latex/ex1_writeup.tex](Latex/ex1_writeup.tex)  
   The source of the report. Read this if you want the exact formulas and wording.

3. [q1_pipeline.py](q1_pipeline.py)  
   The main Q1 pipeline. This is the core modelling file for corner prediction.

4. [team_strength.py](team_strength.py)  
   The Elo-style team-strength updater used inside Q1.

5. [team_variance.py](team_variance.py)  
   The conditional variance model used inside Q1.

6. [market_betting_backtest.py](market_betting_backtest.py)  
   The market-probability layer. It converts Q1 outputs into OU / HC / 1X2 probabilities.

7. [kelly_backtest.py](kelly_backtest.py)  
   The bankroll simulation layer. It applies 100%, 50%, and 25% Kelly under realistic fees and bet caps.

8. [data_cleaning.ipynb](data_cleaning.ipynb)  
   The preprocessing notebook that builds the parquet inputs used by Q1.

9. [Q1.ipynb](Q1.ipynb)  
   A very thin notebook entry point for Q1. It delegates to `q1_pipeline.py`.

## What each file does

### Modelling

- [q1_pipeline.py](q1_pipeline.py)  
  Builds Q1 end to end. It loads the clean parquet files, creates leak-free team features, trains the mean models, estimates variance, builds residual quantiles, and writes the final match-level Q1 predictions.

- [team_strength.py](team_strength.py)  
  Maintains rolling latent team ratings in time order. It estimates how strong each team is at creating corners and how weak it is at preventing them.

- [team_variance.py](team_variance.py)  
  Fits the variance layer used after the mean model. This gives Q1 a spread, not just a point estimate.

- [market_betting_backtest.py](market_betting_backtest.py)  
  Takes Q1 means, variances, and quantile anchors and turns them into market probabilities:
  - OU uses a total-corner distribution
  - HC and 1X2 use a corner-difference distribution  
  It also applies calibration and tail shrink before selection.

- [kelly_backtest.py](kelly_backtest.py)  
  Rebuilds the selected bets from the current Q1 + market backtest layer, then simulates bankroll paths under:
  - initial bankroll `100`
  - fee rate `3%`
  - minimum fee `0.1`
  - maximum single bet `5000`
  - stop if bankroll hits `0`

### Data and notebooks

- [data_cleaning.ipynb](data_cleaning.ipynb)  
  Cleans the raw CSV files and writes:
  - [train.parquet](train.parquet)
  - [betting.parquet](betting.parquet)
  - [all_matches.parquet](all_matches.parquet)

- [Q1.ipynb](Q1.ipynb)  
  A convenience notebook wrapper around `q1_pipeline.py`.

### Input data

- [corners_data.csv](corners_data.csv)  
  Raw match-level corner data.

- [corners_prices.csv](corners_prices.csv)  
  Raw market prices for `1X2`, `OU`, and `HC`.

- [corners_prices_results.csv](corners_prices_results.csv)  
  Realised outcomes used by the betting backtest.

- [train.parquet](train.parquet)  
  Cleaned training table for Q1.

- [betting.parquet](betting.parquet)  
  Cleaned betting-period match table.

- [all_matches.parquet](all_matches.parquet)  
  Combined match history used to construct team-strength features.

### Main outputs

- [q1_betting_match_predictions.csv](q1_betting_match_predictions.csv)  
  The latest Q1 match-level predictions written in the repo root.

- [outputs/q1/q1_betting_match_predictions.csv](outputs/q1/q1_betting_match_predictions.csv)  
  The same Q1 prediction file mirrored into `outputs/q1/`.

- [outputs/kelly_backtest/strategy_summary.csv](outputs/kelly_backtest/strategy_summary.csv)  
  Summary table for 100%, 50%, and 25% Kelly.

- [outputs/kelly_backtest/selected_bets.csv](outputs/kelly_backtest/selected_bets.csv)  
  The final selected bets after calibration, tail shrink, and EV filtering.

- [outputs/kelly_backtest/kelly_100_path.csv](outputs/kelly_backtest/kelly_100_path.csv)  
- [outputs/kelly_backtest/kelly_50_path.csv](outputs/kelly_backtest/kelly_50_path.csv)  
- [outputs/kelly_backtest/kelly_25_path.csv](outputs/kelly_backtest/kelly_25_path.csv)  
  Full realised bankroll paths.

## How to run

### 1. Rebuild clean inputs

Open and run [data_cleaning.ipynb](data_cleaning.ipynb).

This produces:

- `train.parquet`
- `betting.parquet`
- `all_matches.parquet`

### 2. Run Q1

```bash
python q1_pipeline.py --no-plots
```

If you want the diagnostic figures as well:

```bash
python q1_pipeline.py
```

Main Q1 outputs:

- `q1_model_comparison.csv`
- `q1_validation_predictions.csv`
- `q1_betting_match_predictions.csv`
- `outputs/q1/q1_betting_match_predictions.csv`

### 3. Run the Kelly backtest

```bash
python kelly_backtest.py --output-dir outputs/kelly_backtest --initial-bankroll 100 --fee-rate 0.03 --min-fee 0.1 --max-bet 5000
```

This runs:

- `100% Kelly`
- `50% Kelly`
- `25% Kelly`

under the current Q1 predictions and the current market-backtest selection logic.

## Current Kelly assumptions

The current Kelly run uses:

- `1X2` bet only if `EV > 0.03`
- `HC` bet only if `EV > 0.023`
- `OU` bet only if `EV > 0.03`
- initial bankroll = `100`
- fee rate = `3%`
- minimum fee = `0.1`
- maximum stake per bet = `5000`
- stop if bankroll hits `0`

## Where to look for results

If you only want the headline outputs, open these:

1. [Latex/ex1_writeup.pdf](Latex/ex1_writeup.pdf)
2. [outputs/kelly_backtest/strategy_summary.csv](outputs/kelly_backtest/strategy_summary.csv)
3. [outputs/kelly_backtest/cumulative_net_pnl.png](outputs/kelly_backtest/cumulative_net_pnl.png)
4. [outputs/kelly_backtest/bankroll_paths_log.png](outputs/kelly_backtest/bankroll_paths_log.png)
5. [outputs/kelly_backtest/selection_edge_monte_carlo.png](outputs/kelly_backtest/selection_edge_monte_carlo.png)
6. [outputs/kelly_backtest/probability_construction.png](outputs/kelly_backtest/probability_construction.png)

## Minimal mental model of the repo

The shortest way to think about the codebase is:

1. `data_cleaning.ipynb` prepares the clean match tables.
2. `q1_pipeline.py` predicts home and away corner means, variances, and quantile anchors.
3. `market_betting_backtest.py` converts those Q1 outputs into betting probabilities and selections.
4. `kelly_backtest.py` decides how hard to press the edge with different Kelly fractions.
