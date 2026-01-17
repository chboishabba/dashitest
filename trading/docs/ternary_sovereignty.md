# Ternary Sovereignty & The Diophantine Optimum

## Overview
The "Triadic Trading Stack" is an **involutive ternary machine**. It rejects binary conflations that destroy structural information. A core design principle is **Ternary Sovereignty**: every internal decision must maintain triadic (or higher) state logic to avoid "binary drift."

## Ternary States
The primary execution signal is encoded in a 4-valued epistemic space:

| Symbol | Meaning | Bit Representation (2-bit P/N) |
| :--- | :--- | :--- |
| `+1` | **BUY** (Thesis) | `(P=1, N=0)` |
| `-1` | **SELL** (Antithesis) | `(P=0, N=1)` |
| `0`  | **FLAT** (Synthesis) | `(P=0, N=0)` |
| `⊥`  | **PARADOX** (Paradox/M9 Exit) | `(P=1, N=1)` |

*Note: The **UNKNOWN** state (Epistemic Void) acts as a structural hold, separate from the **PARADOX** (Systemic Collapse) which triggers an immediate exit.*

### Conflation Failure (M4 Corruption)
Collapsing `⊥` (Unknown) and `0` (Flat) into a single `0` state is a "compositional failure." It hides the distinction between **deciding to be flat** and **refusing to decide**.

## The Diophantine Optimum (UTES-5)
For the control plane, we utilize the **UTES-5** standard (Uniform Ternary Encoding Standard).

*   **Logic**: 3 trits packed into 5 bits.
*   **Capacity**: $3^3 = 27$ states in $2^5 = 32$ bit codes.
*   **Waste**: 15.6% (vs ~75% for 1-trit-per-byte).
*   **Special Codes (27-31)**: Used for the 4th symbol (`⊥`), `VOID` (27), and `PARADOX` (28).

## M6 Through-State (Tension)
The system must not jump directly from initial data (M3) to closure (M9). It must inhabit the **M6 (Tension)** state, where systemic incoherence is processed as a "rhythm of debate."
*   **M6**: Two triads resolved, one in dialectical tension.
*   **M9**: A higher-order 'mod' applied over M6 to achieve closure without destroying the "philosophical remainder."

## Implementation Requirements
1.  **Separation of Permission from Position**:
    *   **Permission axis (P)**: Prohibited (-1), Suspended (0), Permitted (+1).
    *   **Market Exposure (X)**: Computed only if $P = +1$.
2.  **Pre-Failure Detection**:
    *   Initialize a scaffolding variable $A \in \{-1, 0, +1\}$.
    *   $A = -1$ (Rotten Market) forces $P = -1$ (Hard Closure).
3.  **Triadic XOR**:
    *   Phase-advance rotation (120°).
    *   Contradictions integrate into structure rather than annihilating.

## Bitwise Logic Fusion for GPU Residency
To make the Triadic Stack **logic-limited instead of memory-bound**, the hot path treats ternary values as **2-bit bitplanes** and runs all GF(3) arithmetic through native bitwise gates:

* **Mapping:** Each trit is stored across a Positive (P) and Negative (N) bitplane.
  * $+1$ (Thesis): $(P=1, N=0)$
  * $-1$ (Antithesis): $(P=0, N=1)$
  * $0$ (Synthesis): $(P=0, N=0)$
  * $⊥$ (Paradox/Systemic Collapse Risk): $(P=1, N=1)$ as a semantic poison.
* **SWAR Density:** A single 64-bit register carries 64 ternary lanes, cutting container waste from ~75% (int8) down to ~21–26%.
* **Triadic XOR (120° rotation):**
  * $P_{out} = (P_a \& \sim P_b \& \sim N_b) | (\sim P_a \& P_b \& \sim N_a) | (N_a \& N_b)$
  * $N_{out} = (N_a \& \sim P_b \& \sim N_b) | (\sim P_a \& \sim N_a \& N_b) | (P_a \& P_b)$
  * These branchless formulas avoid binary reflections and preserve the “memory of contradiction.”
* **P/N to Value:** Arithmetic value is recovered as $v = P - N$, so signed GF(3) arithmetic is a subtraction of bitplanes.

## GPU Status & Structural Hot/Cold Split
The Vulkan compute path already mirrors the CPU reference, providing numerical parity, NaN/Inf squashing, and 6×–80× SWAR speedups over scalar ternary logic (1000×+ vs. instruction-level emulators). Heavy math now lives in GPU shaders (`qfeat.comp`), leaving only scalar legitimacy values to cross the CPU/GPU boundary.

* **Hot (Compute):** P/N bitplanes run iterative lattice updates, triadic submissions, and GF(3) rotations.
* **Cold (Storage):** Radix-packed “Large Trytes” (e.g., 9 trits in 16 bits) persist snapshots, hashes, and SVO keys with ~5–12% waste.
* **Transition Policy:** Re-packing to Large Trytes happens only after the reasoning field reaches a stable Region of Convergence (ROC) to respect Landauer limits.
* **Next Milestone:** SVO mask evolution + GF(3) SWAR updates should keep logic entirely within the packed bitplane domain and remove format-shifting entropy losses.

## Operational TODOs
* **SVO masks:** Evolve the mask logic so it can be expressed directly in the packed P/N planes; it is the entry point for mask-aware GPU kernels and the SVO mask evolution milestone.
* **GF(3) SWAR updates:** Finish the remaining SWAR vectorized operators (adders, multipliers, selectors) so the hot path remains branchless and GPU-ready.
* **Triadic XOR parity:** Verify `signals/triadic_ops.py` uses the bitwise formulas above so `scripts/test_ternary_logic.py` keeps passing and the M6 through-state can rotate into M9 synthesis without stalling.
* **UTES-5 signal codes:** Document and enforce the Non-Archimedean NaNs for VOID/PARADOX/qMETA so the 27-state backbone stays coherent and `triadic_ops.py` propagates severity with “Max Severity Wins.”

## The ABC Machine (Burry Zone)
The system evaluates the joint state of the **ABC Machine** to identify asymmetric opportunities:
*   **A (Expressibility)**: Is the tape meaningful?
*   **B (Others' Constraints)**: Are other actors forced/trapped?
*   **C (Structural Integrity)**: Is the market structurally functional?

**The Burry Zone $(A, B, C) = (+1, +1, -1)$**:
In this state, the market is structurally broken ($C=-1$), but the tape remains expressible ($A=+1$) and others are forced to liquidate ($B=+1$). The system adopts a **TRADE_CONVEX** posture, exploiting systemic failure rather than retreating.
