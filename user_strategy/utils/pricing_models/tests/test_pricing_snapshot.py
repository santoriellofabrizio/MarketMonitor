"""Bit-perfect snapshot tests for the linear/cluster pricing models.

These tests pin the numerical output of `MultiPeriodLinearPricingModel.make_matrix_mult`
and `ClusterPricingModel.predict_prices` to fixed MD5 hashes computed against the
pre-refactor implementation. Any future refactor (caching, vectorisation, etc.)
must keep the float64 byte representation identical.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from user_strategy.utils.pricing_models.PricingModel import (
    ClusterPricingModel,
    MultiPeriodLinearPricingModel,
)


def _array_md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def _frame_signature(df: pd.DataFrame) -> tuple:
    return (
        tuple(df.shape),
        tuple(df.index.astype(str)),
        tuple(df.columns.astype(str)),
        _array_md5(df.values),
    )


@pytest.fixture(scope="module")
def linear_full_inputs():
    rng = np.random.default_rng(seed=42)
    n_targets, n_regressors, n_ts = 50, 20, 100
    targets = [f"T{i:02d}" for i in range(n_targets)]
    regressors = [f"R{i:02d}" for i in range(n_regressors)]
    timestamps = pd.date_range("2024-01-01", periods=n_ts, freq="1min")

    beta_data = rng.normal(0.0, 0.3, size=(n_targets, n_regressors))
    beta_data = beta_data * (rng.random(beta_data.shape) > 0.4)
    beta = pd.DataFrame(beta_data, index=targets, columns=regressors)

    returns = pd.DataFrame(
        rng.normal(0.0, 0.001, size=(n_ts, n_regressors)),
        index=timestamps,
        columns=regressors,
    )
    return beta, returns


@pytest.fixture(scope="module")
def cluster_inputs():
    rng = np.random.default_rng(seed=42)
    # mirror the draw order of `linear_full_inputs` so the cluster fixture is
    # generated from a deterministic, independent stream
    rng.normal(0.0, 0.3, size=(50, 20))
    rng.random(size=(50, 20))
    rng.normal(0.0, 0.001, size=(100, 20))

    n_cluster, n_ts = 25, 100
    cluster_targets = [f"C{i:02d}" for i in range(n_cluster)]
    timestamps = pd.date_range("2024-01-01", periods=n_ts, freq="1min")

    cluster_beta_data = rng.normal(0.0, 0.2, size=(n_cluster, n_cluster))
    cluster_beta_data = cluster_beta_data * (rng.random(cluster_beta_data.shape) > 0.3)
    np.fill_diagonal(cluster_beta_data, 0.0)
    beta = pd.DataFrame(cluster_beta_data, index=cluster_targets, columns=cluster_targets)

    returns = pd.DataFrame(
        rng.normal(0.0, 0.001, size=(n_ts, n_cluster)),
        index=timestamps,
        columns=cluster_targets,
    )
    prices = pd.Series(rng.uniform(50, 150, size=n_cluster), index=cluster_targets)
    correction = pd.Series(rng.uniform(0.5, 1.5, size=n_cluster), index=cluster_targets)
    return beta, returns, prices, correction


def test_linear_make_matrix_mult_snapshot_full(linear_full_inputs):
    beta, returns = linear_full_inputs
    model = MultiPeriodLinearPricingModel(beta=beta, returns=returns)
    model.timestamps = returns.index.tolist()

    out = model.make_matrix_mult(returns)
    sig = _frame_signature(out)

    assert sig[0] == (100, 50)
    assert sig[3] == "e333bec0c9f44be65b206b94a93a2c3c", (
        f"baseline drift detected: got {sig[3]}"
    )


def test_linear_missing_regressors_snapshot(linear_full_inputs):
    beta, returns = linear_full_inputs
    returns_partial = returns.drop(columns=["R03", "R07", "R11"])

    model = MultiPeriodLinearPricingModel(beta=beta, returns=returns_partial)
    model.timestamps = returns_partial.index.tolist()

    out = model.make_matrix_mult(returns_partial)
    sig = _frame_signature(out)

    assert sig[0] == (100, 2)
    assert sig[2] == ("T07", "T37")
    assert sig[3] == "0cba88f47a5a01183286429442dc93b8", (
        f"baseline drift detected: got {sig[3]}"
    )


def test_cluster_predict_prices_snapshot(cluster_inputs):
    beta, returns, prices, correction = cluster_inputs
    model = ClusterPricingModel(
        name="TEST_CLUSTER",
        beta=beta,
        returns=returns,
        cluster_correction=correction,
        disable_warning=True,
    )
    model.timestamps = returns.index.tolist()

    out = model.predict_prices(prices, returns)
    sig = _frame_signature(out)

    assert sig[0] == (100, 25)
    assert sig[3] == "5d419a28e48cf40dd42ffa8a9c7f7891", (
        f"baseline drift detected: got {sig[3]}"
    )
