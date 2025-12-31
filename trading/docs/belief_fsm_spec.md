# Belief FSM Spec (Independent Long/Short Chains)

This spec replaces the single thesis FSM with two independent belief chains.
Reference context: `TRADER_CONTEXT.md:105463`, `TRADER_CONTEXT.md:105561`,
`TRADER_CONTEXT.md:105802`, `TRADER_CONTEXT.md:105900`.

## 1) Belief states (per side)

For each side d in {+1,-1}, maintain:

- B_d in {-1, 0, 1, 2}
  - -1 = UNKNOWN (epistemic collapse)
  - 0 = DISBELIEF
  - 1 = WEAK belief
  - 2 = STRONG belief

State: (B_plus, B_minus)

## 2) Evidence signals

Directional (per side):

- alpha(d) in {-1,0,+1}
  - plane_sign = +1 -> alpha(+)=+1, alpha(-)=-1
  - plane_sign = 0 -> both 0

Global (shared):

- beta in {-1,0,+1} (stability)
- rho in {-1,0,+1} (risk)
- pi in {-1,0,+1} (permission)

## 3) Update rule

For each side:

delta = clip_T(alpha(d) + beta + rho + pi)

Update:

1) If pi == -1 -> B_d := 0
2) If alpha == 0 and beta == -1 -> B_d := -1
3) If B_d == -1 -> B_d := max(0, delta)
4) Else -> B_d := clip_0_2(B_d + delta)

## 4) Derived thesis and action

Derived thesis:

- thesis_d := +1 if B_plus > B_minus and B_plus >= 1
- thesis_d := -1 if B_minus > B_plus and B_minus >= 1
- thesis_d := 0 otherwise
- thesis_s := max(B_plus, B_minus)

Action projection:

- action = +1 if B_plus == 2 and B_plus > B_minus
- action = -1 if B_minus == 2 and B_minus > B_plus
- action = 0 otherwise

## 5) Logging (required)

- belief_plus, belief_minus (encode UNKNOWN as -1)
- belief_delta_plus, belief_delta_minus
- belief_state derived from (B_plus, B_minus):
  - UNK if (-1,-1)
  - FLAT if (0,0)
  - L1 if B_plus=1 and B_minus<=1
  - L2 if B_plus=2 and B_plus>B_minus
  - S1 if B_minus=1 and B_plus<=1
  - S2 if B_minus=2 and B_minus>B_plus
  - CONFLICT if (2,2)

## 6) Plot conditioning

Simplex conditioning should support:

- by belief_state (UNK/FLAT/L1/L2/S1/S2/CONFLICT)
- optionally by full lattice (B_plus, B_minus)
