import numpy as np

# ============================================================
# 1) Synthetic 2D single-mode wave task (simple but spectral)
# ============================================================
def make_single_wave(n_points=400, noise=0.03, k_max=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, size=(n_points, 2))

    # Choose one nonzero integer wavevector k
    while True:
        k = rng.integers(-k_max, k_max + 1, size=(2,))
        if not (k[0] == 0 and k[1] == 0):
            break

    phi = rng.uniform(0, 2*np.pi)
    y = np.sin(X @ k + phi) + noise * rng.normal(size=(n_points,))
    return X, y, k, phi


# ============================================================
# 2) Wave / unitary inference over a hypothesis grid Ω
#
#    ψ is complex amplitude over hypotheses (k, φ)
#    Constraints kick phase: ψ <- exp(-i β loss) ψ
#    Optional unitary mixing: FFT diffusion on hypothesis grid
#    Measurement: p = |ψ|^2
# ============================================================
class UnitaryWaveInference:
    def __init__(
        self,
        k_range=(-4, 4),          # inclusive integer range for each component of k
        n_phi=32,                  # number of phase bins in [0, 2π)
        beta=6.0,                  # phase-kick strength (bigger = more aggressive)
        mix_strength=0.15,         # 0..1  (0 = no mixing)
        mix_steps=1,               # how many mixing applications per observation
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)

        self.k_min, self.k_max = k_range
        ks = []
        for k1 in range(self.k_min, self.k_max + 1):
            for k2 in range(self.k_min, self.k_max + 1):
                if k1 == 0 and k2 == 0:
                    continue
                ks.append((k1, k2))
        self.ks = np.array(ks, dtype=int)         # shape (K, 2)
        self.K = self.ks.shape[0]

        self.n_phi = int(n_phi)
        self.phis = np.linspace(0.0, 2*np.pi, self.n_phi, endpoint=False)

        self.beta = float(beta)
        self.mix_strength = float(mix_strength)
        self.mix_steps = int(mix_steps)

        # ψ(k_index, phi_index)
        # Start uniform phase-randomized superposition (still flat in |ψ|^2).
        psi = np.ones((self.K, self.n_phi), dtype=np.complex128)
        psi *= np.exp(1j * self.rng.uniform(0, 2*np.pi, size=psi.shape))
        self.psi = self._normalize(psi)

        # Precompute a fixed unitary "diffusion" operator in Fourier domain:
        # apply FFT -> multiply by phase -> iFFT (all unitary with norm='ortho')
        # We implement it implicitly each time.
        self._make_diffusion_phase()

    def _normalize(self, psi):
        nrm = np.linalg.norm(psi.ravel())
        if nrm == 0:
            # If something went numerically wrong, reset to uniform.
            psi = np.ones_like(psi) / np.sqrt(psi.size)
            return psi.astype(np.complex128)
        return psi / nrm

    def _make_diffusion_phase(self):
        # Build a smooth phase mask in the FFT domain that acts like gentle mixing.
        # This is unitary because it's exp(i * something real).
        K, P = self.K, self.n_phi
        # Frequency grids for FFT axes
        fk = np.fft.fftfreq(K)[:, None]          # shape (K,1)
        fp = np.fft.fftfreq(P)[None, :]          # shape (1,P)

        # radial-ish frequency
        r2 = (fk**2 + fp**2)
        # Phase advance is stronger at low frequencies (gentle diffusion).
        # You can tune the "shape" without breaking unitarity.
        phase = -2*np.pi * (1.0 - np.exp(-25.0 * r2))

        self.diffusion_phase = np.exp(1j * phase)  # unit modulus => unitary

    def _unitary_mix(self, psi):
        # FFT-based unitary mixing:
        # psi <- iFFT( diffusion_phase * FFT(psi) )
        F = np.fft.fft2(psi, norm="ortho")
        F *= self.diffusion_phase
        psi2 = np.fft.ifft2(F, norm="ortho")
        return psi2

    def _predict_grid(self, x):
        # For a given x in R^2, compute y_hat for every (k, phi) hypothesis
        # y_hat(k,phi) = sin(<k,x> + phi)
        # shape: (K, n_phi)
        dots = self.ks @ x  # (K,)
        return np.sin(dots[:, None] + self.phis[None, :])

    def update(self, x, y_obs):
        """
        Apply one observation as a unitary constraint-kick plus optional mixing.

        Constraint kick is diagonal in hypothesis basis:
            ψ(h) <- exp(-i β (y_hat_h(x) - y)^2) ψ(h)

        This is exactly the "constraints as operators / phase rotations" move.
        """
        y_hat = self._predict_grid(x)
        err2 = (y_hat - y_obs) ** 2

        # Phase kick (unit modulus => unitary)
        kick = np.exp(-1j * self.beta * err2)

        psi = kick * self.psi

        # Optional unitary mixing (keeps reversibility / interference dynamics)
        if self.mix_strength > 0:
            for _ in range(self.mix_steps):
                mixed = self._unitary_mix(psi)
                psi = (1.0 - self.mix_strength) * psi + self.mix_strength * mixed
                # Note: convex-combining two unitary results is NOT strictly unitary.
                # We renormalize after; empirically this is a good "soft mixing" knob.
                # If you want strict unitarity, set mix_strength=1.0 and remove blending.

        self.psi = self._normalize(psi)

    def fit(self, X, y, n_steps=None, shuffle=True):
        idx = np.arange(len(X))
        if shuffle:
            self.rng.shuffle(idx)
        if n_steps is None:
            n_steps = len(X)

        for t in range(n_steps):
            i = idx[t % len(X)]
            self.update(X[i], y[i])

    def prob(self):
        p = np.abs(self.psi) ** 2
        return p / p.sum()

    def map_hypothesis(self):
        p = self.prob()
        flat = np.argmax(p)
        ki, pi = np.unravel_index(flat, p.shape)
        return tuple(self.ks[ki]), float(self.phis[pi]), float(p[ki, pi])

    def sample_hypothesis(self):
        p = self.prob().ravel()
        j = self.rng.choice(p.size, p=p)
        ki, pi = np.unravel_index(j, (self.K, self.n_phi))
        return tuple(self.ks[ki]), float(self.phis[pi]), float(p[j])

    def predict_posterior_mean(self, X):
        """
        Bayesian-ish prediction: E[y(x)] under p(h)=|ψ(h)|^2.
        """
        p = self.prob()  # (K, n_phi)
        out = np.zeros((len(X),), dtype=float)
        for i, x in enumerate(X):
            y_hat = self._predict_grid(x)
            out[i] = float(np.sum(p * y_hat))
        return out


def mse(a, b):
    return float(np.mean((a - b) ** 2))


if __name__ == "__main__":
    # ---- generate data ----
    X, y, k_true, phi_true = make_single_wave(n_points=600, noise=0.03, k_max=4, seed=42)

    rng = np.random.default_rng(123)
    idx = rng.permutation(len(X))
    n_train = 120
    tr, te = idx[:n_train], idx[n_train:]

    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # ---- wave learner ----
    model = UnitaryWaveInference(
        k_range=(-4, 4),
        n_phi=48,
        beta=10.0,
        mix_strength=1.0,   # set 1.0 for strictly unitary mixing (no blending)
        mix_steps=1,
        seed=0,
    )

    model.fit(Xtr, ytr, n_steps=2000, shuffle=True)

    k_map, phi_map, p_map = model.map_hypothesis()
    print("TRUE k,phi:", tuple(k_true), float(phi_true))
    print("MAP  k,phi:", k_map, phi_map, "prob≈", p_map)

    yhat = model.predict_posterior_mean(Xte)
    print("Posterior-mean test MSE:", mse(yhat, yte))
