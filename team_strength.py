"""Sequential Bayesian Poisson team-strength estimator.

For each team we maintain four latent skills, all in log-corner space:

    alpha_at_home      -- attack rating when playing at home
    delta_at_home      -- defensive leakiness when playing at home
    alpha_at_away      -- attack rating when playing away
    delta_at_away      -- defensive leakiness when playing away

A general (venue-pooled) attack and defence rating are also maintained:

    alpha              -- attack rating (any venue)
    delta              -- defensive leakiness (any venue)

The model assumes corners are Poisson with log-additive rates:

    log(lambda_home) = league_log_baseline_home + alpha_home + delta_away
    log(lambda_away) = league_log_baseline_away + alpha_away + delta_home

After observing a match's actual corners we update each latent in the
direction of the log-residual, with a per-team learning rate that
shrinks as the team accumulates a longer history. This is a sequential
SGD interpretation of Bayesian shrinkage to the league prior, in the
spirit of Elo and Dixon-Coles models.

The walker iterates matches in chronological order. Features for a
match are *read* before that match's update is applied, so they only
depend on outcomes strictly before the match -- leak-free by
construction.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PRIOR_STRENGTH: int = 15
"""Number of matches at which a team's own history weights equally with the
league prior. Sets the half-life of the per-team learning-rate decay."""

DEFAULT_MAX_LEARNING_RATE: float = 0.10
"""Initial learning rate, applied when a team has zero prior matches."""

LOG_RESIDUAL_SMOOTHING: float = 0.5
"""Constant added inside log() so that zero corners give a finite residual."""


def compute_league_baselines(matches: pd.DataFrame) -> pd.DataFrame:
    """Per-match league baseline corner rates on the log scale.

    For each (competition_id, season_id), an expanding mean of corners
    from matches strictly before the current row. Filled with the global
    mean for the first match of each (competition, season). Returned on
    the log scale, clipped at log(1) for numerical safety.

    Inputs:
        matches: must include match_id, date_time, competition_id,
                 season_id, home_corners, away_corners.

    Returns DataFrame with columns:
        match_id, league_log_baseline_home, league_log_baseline_away
    """
    sorted_matches = matches.sort_values(['competition_id', 'season_id', 'date_time']).copy()
    grouped = sorted_matches.groupby(['competition_id', 'season_id'])

    baseline_home = grouped['home_corners'].transform(lambda s: s.shift(1).expanding().mean())
    baseline_away = grouped['away_corners'].transform(lambda s: s.shift(1).expanding().mean())

    global_mean_home = matches['home_corners'].mean()
    global_mean_away = matches['away_corners'].mean()
    baseline_home = baseline_home.fillna(global_mean_home).clip(lower=1.0)
    baseline_away = baseline_away.fillna(global_mean_away).clip(lower=1.0)

    return pd.DataFrame({
        'match_id': sorted_matches['match_id'].values,
        'league_log_baseline_home': np.log(baseline_home).values,
        'league_log_baseline_away': np.log(baseline_away).values,
    })


class TeamStrengthEstimator:
    """Holds per-team rating state and updates it from match outcomes.

    Maintains six floats per team (general attack/defence plus four
    venue-specific ratings) and a count of prior matches used for the
    learning-rate decay.

    Math:
        log(lambda_home) = league_log_baseline_home + alpha_home + delta_away
        log(lambda_away) = league_log_baseline_away + alpha_away + delta_home

        After observing actual corners (y_h, y_a):
            home_log_residual = log(y_h + 0.5) - log(lambda_home)
            away_log_residual = log(y_a + 0.5) - log(lambda_away)

        Each latent that contributed to a rate moves by half the
        residual scaled by the team's learning rate, so the rate moves
        by exactly the residual.

        learning_rate(team) =
            max_learning_rate * prior_strength
                / (prior_match_count(team) + prior_strength)
    """

    def __init__(self,
                 prior_strength: int = DEFAULT_PRIOR_STRENGTH,
                 max_learning_rate: float = DEFAULT_MAX_LEARNING_RATE) -> None:
        self.prior_strength = prior_strength
        self.max_learning_rate = max_learning_rate

        self.attack_rating: dict[int, float]      = defaultdict(float)
        self.defence_leakiness: dict[int, float]  = defaultdict(float)
        self.attack_at_home: dict[int, float]     = defaultdict(float)
        self.defence_at_home: dict[int, float]    = defaultdict(float)
        self.attack_at_away: dict[int, float]     = defaultdict(float)
        self.defence_at_away: dict[int, float]    = defaultdict(float)
        self.prior_match_count: dict[int, int]    = defaultdict(int)

    def _learning_rate(self, team_id: int) -> float:
        return (self.max_learning_rate * self.prior_strength
                / (self.prior_match_count[team_id] + self.prior_strength))

    def read_features(self,
                      home_team_id: int,
                      away_team_id: int,
                      league_log_baseline_home: float,
                      league_log_baseline_away: float) -> dict[str, float]:
        """Snapshot ratings for a match before any update is applied.

        Returned columns mirror what the GBT consumes downstream. The
        log-rate fields are computed from the *general* (venue-pooled)
        ratings; venue-specific ratings are returned alongside so the
        GBT can split on either view.
        """
        alpha_home = self.attack_rating[home_team_id]
        delta_home = self.defence_leakiness[home_team_id]
        alpha_away = self.attack_rating[away_team_id]
        delta_away = self.defence_leakiness[away_team_id]

        log_lambda_home = league_log_baseline_home + alpha_home + delta_away
        log_lambda_away = league_log_baseline_away + alpha_away + delta_home

        return {
            'home_attack_rating':      alpha_home,
            'home_defence_leakiness':  delta_home,
            'away_attack_rating':      alpha_away,
            'away_defence_leakiness':  delta_away,
            'home_attack_at_home':     self.attack_at_home[home_team_id],
            'home_defence_at_home':    self.defence_at_home[home_team_id],
            'away_attack_at_away':     self.attack_at_away[away_team_id],
            'away_defence_at_away':    self.defence_at_away[away_team_id],
            'log_lambda_home':         log_lambda_home,
            'log_lambda_away':         log_lambda_away,
            'strength_diff':           (alpha_home - delta_home) - (alpha_away - delta_away),
            'venue_strength_diff':     ((self.attack_at_home[home_team_id] - self.defence_at_home[home_team_id])
                                        - (self.attack_at_away[away_team_id] - self.defence_at_away[away_team_id])),
            'prior_match_count_home':  self.prior_match_count[home_team_id],
            'prior_match_count_away':  self.prior_match_count[away_team_id],
            'league_log_baseline_home': league_log_baseline_home,
            'league_log_baseline_away': league_log_baseline_away,
        }

    def apply_match_outcome(self,
                            home_team_id: int,
                            away_team_id: int,
                            home_corners: float,
                            away_corners: float,
                            league_log_baseline_home: float,
                            league_log_baseline_away: float) -> None:
        """Move ratings toward the observed corner counts.

        Updates four general latents (home/away attack, home/away
        defence) and four venue-specific latents (home team's at-home,
        away team's at-away). Increments the prior match count for both
        teams.
        """
        log_lambda_home_general = (league_log_baseline_home
                                   + self.attack_rating[home_team_id]
                                   + self.defence_leakiness[away_team_id])
        log_lambda_away_general = (league_log_baseline_away
                                   + self.attack_rating[away_team_id]
                                   + self.defence_leakiness[home_team_id])

        home_log_residual = np.log(home_corners + LOG_RESIDUAL_SMOOTHING) - log_lambda_home_general
        away_log_residual = np.log(away_corners + LOG_RESIDUAL_SMOOTHING) - log_lambda_away_general

        learning_rate_home = self._learning_rate(home_team_id)
        learning_rate_away = self._learning_rate(away_team_id)

        # General (venue-pooled) ratings: each rate is alpha + delta, so
        # split the residual evenly so that both latents move by half
        # and the sum (the rate) moves by the full residual.
        self.attack_rating[home_team_id]     += 0.5 * learning_rate_home * home_log_residual
        self.defence_leakiness[away_team_id] += 0.5 * learning_rate_away * home_log_residual
        self.attack_rating[away_team_id]     += 0.5 * learning_rate_away * away_log_residual
        self.defence_leakiness[home_team_id] += 0.5 * learning_rate_home * away_log_residual

        # Venue-specific ratings: home team's at-home pair updates from
        # this match (they played at home), away team's at-away pair
        # updates from this match (they played away). The other team's
        # ratings at the *opposite* venue are untouched.
        self.attack_at_home[home_team_id]    += 0.5 * learning_rate_home * home_log_residual
        self.defence_at_home[home_team_id]   += 0.5 * learning_rate_home * away_log_residual
        self.attack_at_away[away_team_id]    += 0.5 * learning_rate_away * away_log_residual
        self.defence_at_away[away_team_id]   += 0.5 * learning_rate_away * home_log_residual

        self.prior_match_count[home_team_id] += 1
        self.prior_match_count[away_team_id] += 1


def walk_matches(matches: pd.DataFrame,
                 target_match_ids: Iterable[int],
                 prior_strength: int = DEFAULT_PRIOR_STRENGTH,
                 max_learning_rate: float = DEFAULT_MAX_LEARNING_RATE) -> pd.DataFrame:
    """Walk every match chronologically, emitting features for targets.

    Iterates `matches` in (date_time, match_id) order. For each match:
    - if it is in `target_match_ids`, snapshot the current ratings as
      features (read before update);
    - if its corners are observed (non-NaN), apply the update.

    Inputs:
        matches: DataFrame with at least match_id, date_time,
                 home_team_id, away_team_id, competition_id, season_id,
                 home_corners, away_corners. The walker uses this set
                 both to drive updates and as the universe of matches
                 to iterate; matches with NaN corners are read-only.
        target_match_ids: set of match_ids for which to emit features.

    Returns one row per target match_id with the columns from
    TeamStrengthEstimator.read_features plus 'match_id'.
    """
    target_set = set(int(m) for m in target_match_ids)

    baselines = compute_league_baselines(matches).set_index('match_id')
    estimator = TeamStrengthEstimator(prior_strength, max_learning_rate)

    ordered = matches.sort_values(['date_time', 'match_id']).reset_index(drop=True)
    feature_rows: list[dict] = []

    for row in ordered.itertuples(index=False):
        match_id = int(row.match_id)
        baseline_home = float(baselines.at[match_id, 'league_log_baseline_home'])
        baseline_away = float(baselines.at[match_id, 'league_log_baseline_away'])

        if match_id in target_set:
            features = estimator.read_features(
                int(row.home_team_id), int(row.away_team_id),
                baseline_home, baseline_away,
            )
            features['match_id'] = match_id
            feature_rows.append(features)

        if not (pd.isna(row.home_corners) or pd.isna(row.away_corners)):
            estimator.apply_match_outcome(
                int(row.home_team_id), int(row.away_team_id),
                float(row.home_corners), float(row.away_corners),
                baseline_home, baseline_away,
            )

    return pd.DataFrame(feature_rows)
