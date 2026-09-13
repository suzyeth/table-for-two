"""Temporal ensembling of ACT action chunks (inference-time only, no retraining)."""
import numpy as np
import pytest

from policy.ov_policy import TemporalEnsemble


def chunk(base, n=4, dim=2):
    """Chunk whose k-th action is base + k, so we can tell which slot was read."""
    return np.stack([np.full(dim, base + k, dtype=np.float32) for k in range(n)])


def test_single_chunk_is_read_in_order():
    ens = TemporalEnsemble(m=0.01)
    ens.add(chunk(10.0))
    assert np.allclose(ens.action(), 10.0)
    ens.step()
    assert np.allclose(ens.action(), 11.0)


def test_equal_weights_average_when_m_is_zero():
    ens = TemporalEnsemble(m=0.0)
    ens.add(chunk(0.0))      # predicts 0,1,2,3 for steps 0..3
    ens.step()
    ens.add(chunk(100.0))    # predicts 100,101,... for steps 1..4
    # step 1: old chunk says 1, new chunk says 100 -> mean 50.5
    assert np.allclose(ens.action(), 50.5)


def test_oldest_prediction_gets_weight_one():
    m = 0.5
    ens = TemporalEnsemble(m=m)
    ens.add(chunk(0.0))
    ens.step()
    ens.add(chunk(100.0))
    w_old, w_new = 1.0, np.exp(-m)  # ACT: w_i = exp(-m * i), i = 0 is the oldest
    expected = (w_old * 1.0 + w_new * 100.0) / (w_old + w_new)
    assert np.allclose(ens.action(), expected)


def test_expired_chunks_are_dropped():
    ens = TemporalEnsemble(m=0.0)
    ens.add(chunk(0.0, n=2))  # covers steps 0 and 1 only
    ens.step(); ens.step()
    ens.add(chunk(7.0, n=2))
    assert np.allclose(ens.action(), 7.0)
    assert len(ens.chunks) == 1


def test_action_without_chunk_raises():
    with pytest.raises(RuntimeError):
        TemporalEnsemble(m=0.01).action()
