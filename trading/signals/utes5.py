"""
UTES-5: Uniform Ternary Encoding Standard (5-bit)
Packing 3 trits into 5 bits (27 states in 32 codes).
Entropy waste: 15.6%.
Special Codes (27-31): VOID, PARADOX, etc.
"""

import numpy as np

# Special Codes
VOID = 27
PARADOX = 28
TAG_30 = 30
TAG_31 = 31

def pack_trits(t1, t2, t3):
    """
    Pack 3 trits (each in {0, 1, 2}) into one 5-bit integer.
    Trits are mapped: -1 -> 0, 0 -> 1, 1 -> 2.
    """
    return int(t1 + 3 * t2 + 9 * t3)

def unpack_trits(packed):
    """
    Unpack a 5-bit integer into 3 trits.
    """
    if packed >= 27:
        return None, None, None  # Special codes
    
    t1 = packed % 3
    t2 = (packed // 3) % 3
    t3 = (packed // 9) % 3
    return t1, t2, t3

def encode_ternary(val):
    """Map -1, 0, 1 to 0, 1, 2"""
    return val + 1

def decode_ternary(val):
    """Map 0, 1, 2 to -1, 0, 1"""
    return val - 1

class UTES5Buffer:
    """
    A buffer that packs ternary signals into uint8 (using only 5 bits per 3 trits).
    In this implementation, we store multiple packed cells in a numpy array.
    """
    def __init__(self, size):
        self.size = size
        self.data = np.zeros((size + 2) // 3, dtype=np.uint8)

    def set_trits(self, index, t1, t2, t3):
        cell_idx = index // 3
        self.data[cell_idx] = pack_trits(t1, t2, t3)

    def get_trits(self, index):
        cell_idx = index // 3
        return unpack_trits(self.data[cell_idx])
