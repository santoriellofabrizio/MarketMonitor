"""Tests for the layout/active cache invalidation in the pricing models."""

import numpy as np
import pandas as pd

from user_strategy.utils.pricing_models.PricingModel import (
    ClusterPricingModel,
    MultiPeriodLinearPricingModel,
)


def _make_linear_model(seed: int = 0):
    rng = np.random.default_rng(seed=seed)
    targets = [f"T{i:02d}" for i in range(10)]
    regressors = [f"R{i:02d}" for i in range(5)]
    timestamps = pd.date_range("2024-01-01", periods=20, freq="1min")

    beta = pd.DataFrame(
        rng.normal(0.0, 0.3, size=(10, 5)),
        index=targets,
        columns=regressors,
    )
    returns = pd.DataFrame(
        rng.normal(0.0, 0.001, size=(20, 5)),
        index=timestamps,
        columns=regressors,
    )
    model = MultiPeriodLinearPricingModel(beta=beta, returns=returns)
    model.timestamps = returns.index.tolist()
    return model, returns


def test_layout_cache_hit_on_repeated_call():
    model, returns = _make_linear_model()

    out1 = model.make_matrix_mult(returns)
    key_after_first = model._layout_cache_key
    cache_after_first = model._layout_cache

    out2 = model.make_matrix_mult(returns)

    pd.testing.assert_frame_equal(out1, out2, check_exact=True)
    assert model._layout_cache_key == key_after_first
    assert model._layout_cache is cache_after_first


def test_layout_cache_invalidated_on_set_beta():
    model, returns = _make_linear_model()

    model.make_matrix_mult(returns)
    assert model._layout_cache_key is not None

    rng = np.random.default_rng(seed=99)
    new_beta = pd.DataFrame(
        rng.normal(0.0, 0.5, size=(10, 5)),
        index=model.target_variables,
        columns=model.regressor,
    )
    model.set_beta(new_beta)

    assert model._layout_cache_key is None
    assert model._layout_cache is None

    out_after = model.make_matrix_mult(returns)
    expected = pd.DataFrame(
        new_beta.values @ returns.reindex(index=model.timestamps).values.T,
        index=new_beta.index,
        columns=model.timestamps,
    ).T
    pd.testing.assert_frame_equal(
        out_after.astype(float), expected.astype(float), check_exact=False, atol=1e-12
    )


def test_layout_cache_invalidated_on_returns_columns_change():
    model, returns = _make_linear_model()

    model.make_matrix_mult(returns)
    cached_active = list(model._layout_cache["active_targets"])
    cached_available = list(model._layout_cache["available_regressors"])

    returns_dropped = returns.drop(columns=["R02"])
    model.make_matrix_mult(returns_dropped)

    assert "R02" not in model._layout_cache["available_regressors"]
    assert model._layout_cache["available_regressors"] != cached_available
    # at least one target had non-zero beta on R02 with seed=0; the set should
    # therefore be a proper subset of the original
    assert len(model._layout_cache["active_targets"]) <= len(cached_active)


def test_layout_cache_invalidated_on_timestamps_change():
    model, returns = _make_linear_model()

    out1 = model.make_matrix_mult(returns)
    key1 = model._layout_cache_key

    shorter = returns.iloc[:10]
    model.timestamps = shorter.index.tolist()
    out2 = model.make_matrix_mult(shorter)

    assert model._layout_cache_key != key1
    assert out2.shape[0] == 10
    pd.testing.assert_frame_equal(
        out2.reset_index(drop=True),
        out1.iloc[:10].reset_index(drop=True),
        check_exact=True,
    )


def test_cluster_active_cache_invalidated_on_set_beta():
    rng = np.random.default_rng(seed=7)
    targets = [f"C{i:02d}" for i in range(8)]
    timestamps = pd.date_range("2024-01-01", periods=20, freq="1min")
    beta_arr = rng.normal(0.0, 0.2, size=(8, 8))
    np.fill_diagonal(beta_arr, 0.0)
    beta = pd.DataFrame(beta_arr, index=targets, columns=targets)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.001, size=(20, 8)),
        index=timestamps,
        columns=targets,
    )
    prices = pd.Series(rng.uniform(50, 150, size=8), index=targets)

    model = ClusterPricingModel(
        name="C", beta=beta, returns=returns, disable_warning=True,
    )
    model.timestamps = returns.index.tolist()

    model.predict_prices(prices, returns)
    assert model._active_cache_key is not None
    assert model._active_cache is not None

    model.set_beta(beta)  # same beta, but `set_beta` must still invalidate
    assert model._active_cache_key is None
    assert model._active_cache is None
    assert model._layout_cache_key is None
