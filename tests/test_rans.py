import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from compression import rans


def test_rans_roundtrip_small():
    data = np.array([0, 1, 1, 2, 2, 2, 3, 255, 0, 0, 128], dtype=np.uint8)
    enc = rans.encode(data)
    dec = rans.decode(enc, length=len(data))
    assert np.array_equal(dec, data)


def test_rans_roundtrip_random():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=500, dtype=np.uint8)
    enc = rans.encode(data)
    dec = rans.decode(enc, length=len(data))
    assert np.array_equal(dec, data)
