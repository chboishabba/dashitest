"""
Sanity checks for thesis memory FSM and benchmark-regret reward.
Run with: python test_thesis_memory.py
"""

try:
    from trading import run_trader
except ModuleNotFoundError:
    import run_trader


def assert_equal(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected}, got {actual}")


def test_entry_requires_promote():
    state = run_trader.ThesisState(d=0, s=0, a=0, c=0, v=0)
    inputs = run_trader.ThesisInputs(
        plane_sign=1,
        plane_sign_flips_w=0,
        plane_would_veto=0,
        stress=0.0,
        p_bad=0.0,
        shadow_would_promote=0,
    )
    params = run_trader.ThesisParams(
        a_max=5,
        cooldown=3,
        p_bad_lo=0.2,
        p_bad_hi=0.7,
        stress_lo=0.2,
        stress_hi=0.7,
    )
    new_state, _, event = run_trader.step_thesis_memory(state, inputs, params)
    assert_equal(new_state.d, 0, "no entry when promote=0")
    assert_equal(event.event, "thesis_none", "event should be none")


def test_entry_happens_on_promote():
    state = run_trader.ThesisState(d=0, s=0, a=0, c=0, v=0)
    inputs = run_trader.ThesisInputs(
        plane_sign=1,
        plane_sign_flips_w=0,
        plane_would_veto=0,
        stress=0.0,
        p_bad=0.0,
        shadow_would_promote=1,
    )
    params = run_trader.ThesisParams(
        a_max=5,
        cooldown=3,
        p_bad_lo=0.2,
        p_bad_hi=0.7,
        stress_lo=0.2,
        stress_hi=0.7,
    )
    new_state, _, event = run_trader.step_thesis_memory(state, inputs, params)
    assert_equal(new_state.d, 1, "entry should set direction")
    assert_equal(new_state.s, 1, "entry should set strength")
    assert_equal(event.event, "thesis_enter", "entry event")


def test_cooldown_blocks_entry():
    state = run_trader.ThesisState(d=0, s=0, a=0, c=2, v=0)
    inputs = run_trader.ThesisInputs(
        plane_sign=1,
        plane_sign_flips_w=0,
        plane_would_veto=0,
        stress=0.0,
        p_bad=0.0,
        shadow_would_promote=1,
    )
    params = run_trader.ThesisParams(
        a_max=5,
        cooldown=3,
        p_bad_lo=0.2,
        p_bad_hi=0.7,
        stress_lo=0.2,
        stress_hi=0.7,
    )
    new_state, _, _ = run_trader.step_thesis_memory(state, inputs, params)
    assert_equal(new_state.d, 0, "cooldown blocks entry")
    assert_equal(new_state.c, 1, "cooldown decrements")


def test_invalidation_exits_fast():
    state = run_trader.ThesisState(d=1, s=1, a=0, c=0, v=1)
    inputs = run_trader.ThesisInputs(
        plane_sign=-1,
        plane_sign_flips_w=0,
        plane_would_veto=0,
        stress=0.9,
        p_bad=0.9,
        shadow_would_promote=0,
    )
    params = run_trader.ThesisParams(
        a_max=5,
        cooldown=3,
        p_bad_lo=0.2,
        p_bad_hi=0.7,
        stress_lo=0.2,
        stress_hi=0.7,
    )
    new_state, _, event = run_trader.step_thesis_memory(state, inputs, params)
    assert_equal(new_state.d, 0, "invalidated thesis exits")
    assert_equal(new_state.c, 3, "cooldown set on exit")
    assert_equal(event.event, "thesis_exit", "exit event")
    assert_equal(event.reason, "thesis_invalidated", "exit reason")


def test_strength_clips():
    state = run_trader.ThesisState(d=1, s=2, a=0, c=0, v=0)
    inputs = run_trader.ThesisInputs(
        plane_sign=1,
        plane_sign_flips_w=0,
        plane_would_veto=0,
        stress=0.0,
        p_bad=0.0,
        shadow_would_promote=0,
    )
    params = run_trader.ThesisParams(
        a_max=5,
        cooldown=3,
        p_bad_lo=0.2,
        p_bad_hi=0.7,
        stress_lo=0.2,
        stress_hi=0.7,
    )
    new_state, _, event = run_trader.step_thesis_memory(state, inputs, params)
    assert_equal(new_state.s, 2, "strength clamps at 2")
    assert_equal(event.event, "thesis_update", "update event")


def test_no_flipflop():
    derived = run_trader.ThesisDerived(alpha=1, beta=1, rho=1, ds=1, sum=3)
    action_t, override = run_trader.apply_thesis_constraints(
        thesis_d=1,
        derived=derived,
        proposed_action=-1,
        hard_veto=False,
        exit_trigger=False,
    )
    assert_equal(action_t, 1, "flip-flop overridden to thesis direction")
    assert_equal(override, "thesis_no_flipflop", "override reason")

    derived_bad = run_trader.ThesisDerived(alpha=-1, beta=-1, rho=-1, ds=-1, sum=-3)
    action_t, override = run_trader.apply_thesis_constraints(
        thesis_d=1,
        derived=derived_bad,
        proposed_action=-1,
        hard_veto=False,
        exit_trigger=False,
    )
    assert_equal(action_t, 0, "flip-flop de-risks under bad evidence")
    assert_equal(override, "thesis_no_flipflop", "override reason")


def test_regret_reward_penalizes_flat():
    reward = run_trader.compute_regret_reward(0, 0.01, 1.0, 0.0)
    if reward >= 0:
        raise AssertionError("flat should be penalized when benchmark is long and return is positive")


def main():
    test_entry_requires_promote()
    test_entry_happens_on_promote()
    test_cooldown_blocks_entry()
    test_invalidation_exits_fast()
    test_strength_clips()
    test_no_flipflop()
    test_regret_reward_penalizes_flat()
    print("PASS: thesis memory FSM tests.")


if __name__ == "__main__":
    main()
