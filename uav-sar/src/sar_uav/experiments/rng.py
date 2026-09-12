"""Indexed 4-tuple RNG stream for Common Random Numbers (CRN).

Corresponds to proposal section "Protocol danh gia":
    "Su dung ky thuat Common Random Numbers (CRN) voi co che sinh so ngau nhien
     duoc lap chi muc (indexed RNG stream) gan chat theo bo 4 chi muc xac dinh:
     (mission_instance, thoi diem tick t, UAV k, loai su kien cam bien/chuyen dong)"
"""

from __future__ import annotations

import hashlib
import struct
from typing import Tuple
import numpy as np


class IndexedRNGStream:
    """Provides deterministic, independent pseudo-random numbers keyed by 4-tuple:
    (mission_idx, tick, uav_idx, event_type).
    """

    def __init__(self, master_seed: int = 42):
        self.master_seed = int(master_seed)

    def _hash_to_uint32(self, mission_idx: int, tick: int, uav_idx: int, event_type: str) -> int:
        key_str = f"{self.master_seed}:{mission_idx}:{tick}:{uav_idx}:{event_type}"
        h = hashlib.sha256(key_str.encode("utf-8")).digest()
        val = struct.unpack(">I", h[:4])[0]
        return val

    def get_rng(self, mission_idx: int, tick: int, uav_idx: int, event_type: str) -> np.random.RandomState:
        seed = self._hash_to_uint32(mission_idx, tick, uav_idx, event_type)
        return np.random.RandomState(seed)

    def uniform(self, mission_idx: int, tick: int, uav_idx: int, event_type: str) -> float:
        rng = self.get_rng(mission_idx, tick, uav_idx, event_type)
        return float(rng.uniform(0.0, 1.0))
