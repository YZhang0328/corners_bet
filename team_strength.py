"""Sequential team-strength estimator for football corners.

The goal is simple:

- keep a running estimate of how good each team is at *winning* corners;
- keep a running estimate of how likely each team is to *allow* corners;
- update those estimates match by match in time order;
- only expose pre-match information to the downstream Q1 model.

For each team we maintain four venue-specific latent skills, all in
log-corner space:

    alpha_at_home      -- attack rating when playing at home
    delta_at_home      -- defensive leakiness when playing at home
    alpha_at_away      -- attack rating when playing away
    delta_at_away      -- defensive leakiness when playing away

A general (venue-pooled) attack and defence rating are also maintained:

    alpha              -- attack rating (any venue)
    delta              -- defensive leakiness (any venue)

The core assumption is that expected corner intensity is log-additive:

    log(lambda_home) = league_log_baseline_home + alpha_home + delta_away
    log(lambda_away) = league_log_baseline_away + alpha_away + delta_home

After a match finishes, we compare:

- what the model expected before kickoff; and
- what actually happened.

If a team wins more corners than expected, its attack rating moves up.
If a team allows more opponent corners than expected, its defensive
leakiness moves up. Teams with little history move faster. Teams with
long history move more slowly. This is an Elo-style online update,
adapted from win/loss ratings to corner production and concession.

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
"""Controls how quickly a new team's rating can move away from the league prior."""

DEFAULT_MAX_LEARNING_RATE: float = 0.10
"""Largest update size used when a team has almost no prior history."""

LOG_RESIDUAL_SMOOTHING: float = 0.5
"""Small smoothing constant so that zero corners still produce a finite log residual."""


def compute_league_baselines(matches: pd.DataFrame) -> pd.DataFrame:
    """Build the running league baseline before each match."""
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
    """Store team ratings and update them after each match.

    Each team has attack and defensive-leakiness ratings, both overall
    and split by venue. These ratings are learned online from past
    matches. If a team wins more corners than expected, its attack
    rating goes up. If it allows more corners than expected, its
    defensive leakiness goes up.
    """

    def __init__(self,
                 prior_strength: int = DEFAULT_PRIOR_STRENGTH,
                 max_learning_rate: float = DEFAULT_MAX_LEARNING_RATE) -> None:
        self.prior_strength = prior_strength
        self.max_learning_rate = max_learning_rate

        self.overall_attack_rating: dict[int, float] = defaultdict(float)
        self.overall_defence_leakiness: dict[int, float] = defaultdict(float)
        self.home_attack_rating: dict[int, float] = defaultdict(float)
        self.home_defence_leakiness: dict[int, float] = defaultdict(float)
        self.away_attack_rating: dict[int, float] = defaultdict(float)
        self.away_defence_leakiness: dict[int, float] = defaultdict(float)
        self.prior_match_count: dict[int, int] = defaultdict(int)

    def _learning_rate(self, team_id: int) -> float:
        return (self.max_learning_rate * self.prior_strength
                / (self.prior_match_count[team_id] + self.prior_strength))

    def read_features(self,
                      home_team_id: int,
                      away_team_id: int,
                      league_log_baseline_home: float,
                      league_log_baseline_away: float) -> dict[str, float]:
        """Read the team's current ratings before this match is learned from.

        This is the leak-free pre-match snapshot that Q1 consumes.
        Think of it as:

        - what do we believe about the home team right now?
        - what do we believe about the away team right now?
        - if the match started now, what corner rates would those
          beliefs imply?

        The returned columns are exactly the features used downstream by
        the gradient-boosted tree model in Q1.

        Two views are returned:

        - general ratings, pooled across all venues;
        - venue-specific ratings, so Q1 can learn separate home and
          away effects if that helps.
        """
        home_team_overall_attack = self.overall_attack_rating[home_team_id]
        home_team_overall_defence = self.overall_defence_leakiness[home_team_id]
        away_team_overall_attack = self.overall_attack_rating[away_team_id]
        away_team_overall_defence = self.overall_defence_leakiness[away_team_id]

        predicted_home_log_rate = league_log_baseline_home + home_team_overall_attack + away_team_overall_defence
        predicted_away_log_rate = league_log_baseline_away + away_team_overall_attack + home_team_overall_defence

        return {
            'home_attack_rating':      home_team_overall_attack,
            'home_defence_leakiness':  home_team_overall_defence,
            'away_attack_rating':      away_team_overall_attack,
            'away_defence_leakiness':  away_team_overall_defence,
            'home_attack_at_home':     self.home_attack_rating[home_team_id],
            'home_defence_at_home':    self.home_defence_leakiness[home_team_id],
            'away_attack_at_away':     self.away_attack_rating[away_team_id],
            'away_defence_at_away':    self.away_defence_leakiness[away_team_id],
            'log_lambda_home':         predicted_home_log_rate,
            'log_lambda_away':         predicted_away_log_rate,
            'strength_diff':           (home_team_overall_attack - home_team_overall_defence) - (away_team_overall_attack - away_team_overall_defence),
            'venue_strength_diff':     ((self.home_attack_rating[home_team_id] - self.home_defence_leakiness[home_team_id])
                                        - (self.away_attack_rating[away_team_id] - self.away_defence_leakiness[away_team_id])),
            'prior_match_count_home': self.prior_match_count[home_team_id],
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
        """Update team ratings after the match result is known.

        Plain-language version:

        - first compute what the model expected before kickoff;
        - compare that with the actual home and away corner counts;
        - if a team did better than expected, strengthen the ratings
          that helped produce that outcome;
        - if a team did worse than expected, weaken those ratings.

        Example:

        - if the home team wins more corners than expected, increase the
          home attack rating and increase the away defensive leakiness;
        - if the away team wins more corners than expected, increase the
          away attack rating and increase the home defensive leakiness.

        We update both the general ratings and the venue-specific
        ratings, then increment each team's match count so future
        updates become a bit smaller.
        """
        predicted_home_log_rate = (
            league_log_baseline_home
            + self.overall_attack_rating[home_team_id]
            + self.overall_defence_leakiness[away_team_id]
        )
        predicted_away_log_rate = (
            league_log_baseline_away
            + self.overall_attack_rating[away_team_id]
            + self.overall_defence_leakiness[home_team_id]
        )

        home_team_log_residual = np.log(home_corners + LOG_RESIDUAL_SMOOTHING) - predicted_home_log_rate
        away_team_log_residual = np.log(away_corners + LOG_RESIDUAL_SMOOTHING) - predicted_away_log_rate

        home_team_learning_rate = self._learning_rate(home_team_id)
        away_team_learning_rate = self._learning_rate(away_team_id)

        # General ratings:
        # home corners above expectation -> raise home attack and away leakiness
        # away corners above expectation -> raise away attack and home leakiness
        # We split each residual in half so the attack side and defence side
        # share the update.
        self.overall_attack_rating[home_team_id] += 0.5 * home_team_learning_rate * home_team_log_residual
        self.overall_defence_leakiness[away_team_id] += 0.5 * away_team_learning_rate * home_team_log_residual
        self.overall_attack_rating[away_team_id] += 0.5 * away_team_learning_rate * away_team_log_residual
        self.overall_defence_leakiness[home_team_id] += 0.5 * home_team_learning_rate * away_team_log_residual

        # Venue-specific ratings:
        # this match only teaches us about the home team's "at home"
        # behaviour and the away team's "away" behaviour.
        self.home_attack_rating[home_team_id] += 0.5 * home_team_learning_rate * home_team_log_residual
        self.home_defence_leakiness[home_team_id] += 0.5 * home_team_learning_rate * away_team_log_residual
        self.away_attack_rating[away_team_id] += 0.5 * away_team_learning_rate * away_team_log_residual
        self.away_defence_leakiness[away_team_id] += 0.5 * away_team_learning_rate * home_team_log_residual

        self.prior_match_count[home_team_id] += 1
        self.prior_match_count[away_team_id] += 1


def walk_matches(matches: pd.DataFrame,
                 target_match_ids: Iterable[int],
                 prior_strength: int = DEFAULT_PRIOR_STRENGTH,
                 max_learning_rate: float = DEFAULT_MAX_LEARNING_RATE) -> pd.DataFrame:
    """Walk through matches in time order and emit leak-free team features."""
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
