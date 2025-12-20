"""
optimal_packings_table.py
-------------------------
Generate optimal k-trit packings:
- For k = 1..40, compute minimum bits b_min, spares = 2^b - 3^k, efficiency.
- Filter for byte-aligned containers (b multiple of 8) and per-byte packing (5 trits/byte).
"""

import math


def packings():
    rows = []
    for k in range(1, 41):
        b_min = math.ceil(k * math.log2(3))
        spares = (1 << b_min) - (3 ** k)
        eff = (k * math.log2(3)) / b_min
        rows.append((k, b_min, spares, eff))
    return rows


def main():
    rows = packings()
    print("k  b_min  spares        efficiency")
    for k, b, s, eff in rows:
        print(f"{k:2d} {b:5d} {s:8d} {eff*100:9.2f}%")
    print("\nByte-aligned (b multiple of 8):")
    for k, b, s, eff in rows:
        if b % 8 == 0:
            print(f"{k:2d} {b:5d} {s:8d} {eff*100:9.2f}%")
    print("\nPer-byte optimal packing (5 trits/byte):")
    k_byte = 5
    b_byte = 8
    spares_byte = (1 << b_byte) - (3 ** k_byte)
    eff_byte = (k_byte * math.log2(3)) / b_byte
    print(f"5 trits in 8 bits: spares={spares_byte}, efficiency={eff_byte*100:.2f}% (~95% entropy efficiency)")


if __name__ == "__main__":
    main()
