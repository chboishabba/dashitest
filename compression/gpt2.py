import math

BITS = [8, 16, 32, 64, 128, 256, 512]

LOG2_3 = math.log2(3)

def max_k_for_bits(b: int, spare_codes: int = 0) -> int:
    """
    Max k such that 3^k <= 2^b - spare_codes
    """
    cap = (1 << b) - spare_codes
    if cap <= 0:
        return 0
    # start from the Shannon upper bound
    k = int(math.floor(math.log2(cap) / LOG2_3))
    # adjust to be safe against float rounding
    while pow(3, k) > cap and k > 0:
        k -= 1
    while pow(3, k + 1) <= cap:
        k += 1
    return k

def stats(b: int, k: int):
    cap = 1 << b
    used = pow(3, k)
    spares = cap - used
    info_bits = k * LOG2_3
    eff = info_bits / b if b else 0.0
    waste = 1.0 - eff
    return {
        "bits": b,
        "k_trits": k,
        "states_used": used,
        "states_total": cap,
        "spare_codes": spares,
        "info_bits": info_bits,
        "efficiency": eff,
        "waste": waste,
    }

def run(spare_codes: int):
    print(f"\n=== Optimal packings with spare_codes >= {spare_codes} ===")
    print("bits  k_trits  efficiency   waste     spare_codes")
    for b in BITS:
        k = max_k_for_bits(b, spare_codes=spare_codes)
        s = stats(b, k)
        print(
            f"{s['bits']:>4}  {s['k_trits']:>6}   "
            f"{s['efficiency']*100:>9.3f}%  {s['waste']*100:>7.3f}%  "
            f"{s['spare_codes']:>12}"
        )

if __name__ == "__main__":
    # Choose your traversal budget here:
    # e.g. reserve at least 13 spare codes in each container (mirrors the 5-trits-per-byte spare budget)
    run(spare_codes=0)
    run(spare_codes=13)
    run(spare_codes=256)   # example: reserve a whole "page" of markers
