"""Fit the Q1 variance layer on top of the mean predictions.

Q1 first predicts the centre of each team's corner count.
This file predicts how wide the distribution should be around that centre.

The model is:

    log(variance) =
        intercept
        + beta_1 * predicted_mean
        + beta_2 * market_certainty
        + beta_3 * rolling_std

Then the predicted variance is:

    variance = calibration_scale * exp(log(variance))

The three inputs are:

- predicted_mean: Q1's point prediction for that team;
- market_certainty: how one-sided the market price looks;
- rolling_std: how volatile that team has been recently.

The target is the squared residual:

    (observed_corners - predicted_mean)^2

but it is fitted in log space:

    log((observed_corners - predicted_mean)^2 + offset)

This keeps the fitted variance positive and makes the regression more stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression


VARIANCE_LOG_OFFSET: float = 0.5
"""Constant added inside log() so log((y - mu)^2 + offset) is finite when
y == mu exactly."""


@dataclass
class DispersionModel:
    """Store one fitted log-variance regression and its scale correction."""
    intercept: float
    coefficients: np.ndarray
    calibration_scale: float

    def predict(
        self,
        predicted_mean: np.ndarray,
        market_certainty: np.ndarray,
        rolling_std: np.ndarray,
    ) -> np.ndarray:
        """Turn mean, market certainty, and recent volatility into a variance."""
        design = np.column_stack([predicted_mean, market_certainty, rolling_std])
        log_variance_raw = self.intercept + design @ self.coefficients
        return self.calibration_scale * np.exp(log_variance_raw)


def fit_dispersion(
    predicted_mean: np.ndarray,
    observed_corners: np.ndarray,
    market_certainty: np.ndarray,
    rolling_std: np.ndarray,
    variance_log_offset: float = VARIANCE_LOG_OFFSET,
) -> DispersionModel:
    """Fit one linear model for log residual variance on a calibration sample.

    In plain terms:

    - start from the Q1 point prediction;
    - measure how wrong it was with squared residuals;
    - regress log squared residuals on mean, market certainty, and rolling std;
    - rescale the fitted values so the average predicted variance matches the
      average realised squared residual.
    """
    design = np.column_stack([predicted_mean, market_certainty, rolling_std])
    squared_residuals = (observed_corners - predicted_mean) ** 2
    target = np.log(squared_residuals + variance_log_offset)

    fit = LinearRegression().fit(design, target)
    raw_predicted_variance = np.exp(fit.intercept_ + design @ fit.coef_)
    calibration_scale = float(squared_residuals.mean() / raw_predicted_variance.mean())

    return DispersionModel(
        intercept=float(fit.intercept_),
        coefficients=np.asarray(fit.coef_, dtype=float),
        calibration_scale=calibration_scale,
    )
