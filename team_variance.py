"""Heteroscedastic dispersion model: predict per-match conditional variance.

Stage 2 of the corner pipeline produces a point prediction for each side
(home or away). The realised count scatters around that prediction with a
variance that depends on:

    - the level of the predicted mean
    - market certainty   (heavy 1X2 favourites scatter less)
    - the team's own historical variance behaviour

We model:

    log(sigma^2) = beta_0 + beta_1 * predicted_mean
                          + beta_2 * market_certainty
                          + beta_3 * rolling_std

where `market_certainty = abs(p_team_1x2 - 0.5)` (zero at coin-flip,
0.5 at certainty), fit by ordinary least squares with target

    log( (y - mu)^2 + epsilon )

on a calibration partition. Working in log-space keeps `sigma^2`
positive and matches how variance scales multiplicatively with means
in count distributions.

The raw OLS prediction is biased low because of Jensen's inequality
on log() and the small `epsilon` floor we add for numerical safety. We
correct with a single multiplicative scale chosen so the predicted
variance averages match the realised squared residuals on the
calibration set:

    calibration_scale = mean(squared_residual) / mean(exp(beta . x))

After this scalar correction, predicted-variance mean equals
realised-residual-square mean on the calibration set by construction.
The scale is then frozen on the held-out partition.
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
    """Fitted dispersion model with bias-corrected variance prediction.

    Attributes:
        intercept:          beta_0 from the OLS fit (log-variance scale)
        coefficients:       (beta_1, beta_2, beta_3) for
                            (mu, market_certainty, rolling_std)
        calibration_scale:  multiplicative correction applied to
                            exp(intercept + coefficients . x) so that
                            the mean of predicted sigma^2 on the
                            calibration set matches the mean of
                            realised squared residuals.
    """
    intercept: float
    coefficients: np.ndarray
    calibration_scale: float

    def predict(
        self,
        predicted_mean: np.ndarray,
        market_certainty: np.ndarray,
        rolling_std: np.ndarray,
    ) -> np.ndarray:
        """Predict sigma^2 for each match.

        Inputs are 1D arrays of equal length:
            predicted_mean    -- Stage 2 point prediction for the side
            market_certainty  -- abs(p_1x2 - 0.5), zero at coin-flip
            rolling_std       -- team's rolling-window corner std

        Returns sigma^2 (1D array, same length).
        """
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
    """Fit log-OLS dispersion model and apply the Jensen bias correction.

    Inputs (all 1D arrays of equal length, all on the calibration set):
        predicted_mean    -- Stage 2 point predictions
        observed_corners  -- realised corner counts
        market_certainty  -- abs(p_1x2 - 0.5)
        rolling_std       -- team rolling-window corner std

    Returns a DispersionModel ready for `.predict(...)` on any data.
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
