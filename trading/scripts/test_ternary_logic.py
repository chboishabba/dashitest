import numpy as np
from signals.triadic import BUY, SELL, FLAT, UNKNOWN, PARADOX, compute_triadic_state
from signals.triadic_ops import triadic_xor_bitwise, triadic_step_rotation
from signals.utes5 import UTES5Buffer, pack_trits, unpack_trits

def test_triadic_ops():
    print("Testing Triadic Bitwise Ops (P/N Bitplanes)...")
    # Mapping: +1=(1,0), -1=(0,1), 0=(0,0), ⊥=(1,1)
    
    # 1 + 1 = -1
    p, n = triadic_xor_bitwise(1, 0, 1, 0)
    print(f"1+1: p={p}, n={n}")
    assert (p, n) == (0, 1), f"1+1 expected (0,1), got ({p},{n})"
    
    # PARADOX (1,1) propagation
    p, n = triadic_xor_bitwise(1, 0, 1, 1)
    print(f"1+⊥: p={p}, n={n}")
    assert (p, n) == (1, 1), f"1+⊥ expected (1,1), got ({p},{n})"
    
    # -1 + -1 = 1
    p, n = triadic_xor_bitwise(0, 1, 0, 1)
    print(f"-1-1: p={p}, n={n}")
    assert (p, n) == (1, 0), f"-1-1 expected (1,0), got ({p},{n})"
    
    # 1 + -1 = 0
    p, n = triadic_xor_bitwise(1, 0, 0, 1)
    assert (p, n) == (0, 0), f"1-1 expected (0,0), got ({p},{n})"
    
    # 1 + ⊥ = ⊥
    p, n = triadic_xor_bitwise(1, 0, 1, 1)
    assert (p, n) == (1, 1), f"1+⊥ expected (1,1), got ({p},{n})"
    
    # Rotation (Add +1)
    p, n = 0, 0 # FLAT
    p, n = triadic_step_rotation(p, n) # -> BUY
    print(f"Rot1 (to BUY): p={p}, n={n}")
    assert (p, n) == (1, 0)
    p, n = triadic_step_rotation(p, n) # -> SELL
    print(f"Rot2 (to SELL): p={p}, n={n}")
    assert (p, n) == (0, 1)
    p, n = triadic_step_rotation(p, n) # -> FLAT
    print(f"Rot3 (to FLAT): p={p}, n={n}")
    assert (p, n) == (0, 0)
    
    print("Triadic Ops: PASS")

def test_utes5():
    print("Testing UTES-5 Packing (3 trits / 5 bits)...")
    t1, t2, t3 = 1, 0, 2 # trits in {0,1,2}
    packed = pack_trits(t1, t2, t3)
    assert packed < 27
    u1, u2, u3 = unpack_trits(packed)
    assert (t1, t2, t3) == (u1, u2, u3)
    
    buffer = UTES5Buffer(10)
    buffer.set_trits(0, 2, 2, 1)
    r1, r2, r3 = buffer.get_trits(0)
    assert (r1, r2, r3) == (2, 2, 1)
    
    print("UTES-5: PASS")

def test_triadic_state():
    print("Testing Triadic State (UNKNOWN handling)...")
    prices = [100.0, 101.0, 102.0, np.nan, 103.0]
    state = compute_triadic_state(prices)
    print(f"State: {state}")
    assert state[3] == UNKNOWN
    print("Triadic State: PASS")

def test_strategy_paradox():
    print("Testing Strategy PARADOX (M9 Hard Exit)...")
    from strategy.triadic_strategy import TriadicStrategy
    strat = TriadicStrategy(symbol="TEST")
    strat.position = 0.5 # Assume currently long
    
    intent = strat.step(ts=1000, state=PARADOX)
    print(f"Paradox intent: target={intent.target_exposure}, reason={intent.reason}")
    assert intent.target_exposure == 0.0
    assert intent.hold == False
    assert intent.reason == "systemic_collapse_paradox (⚡)"
    print("Strategy PARADOX: PASS")

if __name__ == "__main__":
    try:
        test_triadic_ops()
        test_utes5()
        test_triadic_state()
        test_strategy_paradox()
        print("\nAll Ternary Sovereignty tests passed!")
    except AssertionError as e:
        import traceback
        print(f"\nAssertion FAILED:")
        traceback.print_exc()
    except Exception as e:
        import traceback
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
