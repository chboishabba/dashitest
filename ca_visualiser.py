import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- CA imports ---
from levin_ca_train import (
    step_grid,
    make_dataset as make_dataset_levin,
    features_from_grid,
    train_logreg as train_logreg_levin,
)
from motif_ca import (
    step_motif_ca,
    features_from_state,
    train_logreg as train_logreg_motif,
    make_dataset as make_dataset_motif,
    MotifParams,
)

# ---------- Learned rule inference ----------
def step_grid_learned(grid: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Apply the learned logistic regression rule to a full grid.
    W: (6,3) weight matrix from train_logreg()
    """
    H, Wd = grid.shape
    feats = features_from_grid(grid)          # (H*W, 6)
    logits = feats @ W                        # (H*W, 3)
    pred = logits.argmax(axis=1).astype(np.int8)
    return pred.reshape(H, Wd)


def step_motif_learned(G: np.ndarray, F: np.ndarray, A: np.ndarray, W: np.ndarray, params: MotifParams):
    """
    Apply the learned logistic regression rule to the motif CA.
    Learner predicts next G; fatigue F is updated deterministically using predicted G.
    """
    H, Wd = G.shape
    feats = features_from_state(G, F, A)  # (H*W, 8)
    logits = feats @ W                    # (H*W, 3)
    pred = logits.argmax(axis=1).astype(np.int8).reshape(H, Wd)
    # fatigue update mirrors true rule's growth/decay
    F_next = F + (pred == 1) * params.fatigue_inc - (pred != 1) * params.fatigue_dec
    F_next = np.clip(F_next, 0, params.fatigue_max)
    return pred, F_next


# ---------- Visualiser ----------
class CAVisualizer:
    def __init__(self, H=128, W=128, seed=0, train_samples=200, show_learned=True, mode="levin"):
        self.rng = np.random.default_rng(seed)
        self.H, self.W = H, W
        self.mode = mode
        self.params = MotifParams() if mode == "motif" else None

        # state
        self.grid = self.rng.integers(0, 3, size=(H, W), dtype=np.int8)
        self.grid_true = self.grid.copy()
        self.grid_learned = self.grid.copy()
        self.fat_true = self.rng.integers(0, 4, size=(H, W), dtype=np.int8) if mode == "motif" else None
        self.fat_learned = self.fat_true.copy() if mode == "motif" else None
        self.anchor = (self.rng.random((H, W)) < (self.params.anchor_prob if self.params else 0.2)).astype(np.int8) if mode == "motif" else None

        # animation controls
        self.paused = False
        self.step_once = False
        self.delay = 0.02  # seconds between frames
        self.show_learned = show_learned
        self.compare = show_learned  # start in compare mode if learned is enabled

        # train a tiny learner to recover the rule (fast)
        self.W_lr = None
        if show_learned:
            self.W_lr = self._train_rule(train_samples=train_samples)

        # plotting
        self.cmap = ListedColormap(["black", "gray", "white"])  # 0,1,2
        self.fig = plt.figure(figsize=(10, 5 if show_learned else 5))
        self._setup_axes()

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _train_rule(self, train_samples=200):
        t0 = time.perf_counter()
        if self.mode == "motif":
            X, Y, _, _ = make_dataset_motif(num_samples=train_samples, H=32, W=32, seed=0, params=self.params)
            W_lr = train_logreg_motif(X, Y, lr=5e-3, iters=400)
        else:
            X, Y = make_dataset_levin(num_samples=train_samples, H=32, W=32, seed=0)
            X_feats = np.concatenate([features_from_grid(x) for x in X], axis=0)
            Y_flat = np.concatenate([y.ravel() for y in Y], axis=0)
            W_lr = train_logreg_levin(X_feats, Y_flat, lr=5e-3, iters=300)
        t1 = time.perf_counter()
        print(f"[learn] trained log-reg in {(t1 - t0) * 1e3:.1f} ms on {train_samples} samples (mode={self.mode})")
        return W_lr

    def _setup_axes(self):
        self.fig.clf()

        if self.show_learned and self.compare:
            self.ax1 = self.fig.add_subplot(1, 2, 1)
            self.ax2 = self.fig.add_subplot(1, 2, 2)

            self.im1 = self.ax1.imshow(self.grid_true, cmap=self.cmap, vmin=0, vmax=2, interpolation="nearest")
            self.im2 = self.ax2.imshow(self.grid_learned, cmap=self.cmap, vmin=0, vmax=2, interpolation="nearest")

            self.ax1.set_title("True CA")
            self.ax2.set_title("Learned CA (log-reg)")
            for ax in (self.ax1, self.ax2):
                ax.set_xticks([])
                ax.set_yticks([])

            self.text = self.fig.text(0.5, 0.01, "", ha="center", va="bottom")
        else:
            self.ax1 = self.fig.add_subplot(1, 1, 1)
            self.im1 = self.ax1.imshow(self.grid_true, cmap=self.cmap, vmin=0, vmax=2, interpolation="nearest")
            self.ax1.set_title("True CA")
            self.ax1.set_xticks([])
            self.ax1.set_yticks([])
            self.im2 = None
            self.text = self.fig.text(0.5, 0.01, "", ha="center", va="bottom")

        self.fig.tight_layout(rect=(0, 0.03, 1, 1))

    def _randomize(self):
        if self.mode == "motif" and self.params:
            self.grid = self.rng.choice(3, size=(self.H, self.W), p=self.params.state_probs).astype(np.int8)
        else:
            self.grid = self.rng.integers(0, 3, size=(self.H, self.W), dtype=np.int8)
        self.grid_true = self.grid.copy()
        self.grid_learned = self.grid.copy()
        if self.mode == "motif":
            self.fat_true = self.rng.integers(0, self.params.fatigue_max + 1, size=(self.H, self.W), dtype=np.int8)
            self.fat_learned = self.fat_true.copy()
            self.anchor = (self.rng.random((self.H, self.W)) < self.params.anchor_prob).astype(np.int8)

    def _step(self):
        if self.mode == "motif":
            self.grid_true, self.fat_true = step_motif_ca(self.grid_true, self.fat_true, self.anchor, self.params)
            if self.show_learned and self.W_lr is not None:
                self.grid_learned, self.fat_learned = step_motif_learned(
                    self.grid_learned, self.fat_learned, self.anchor, self.W_lr, self.params
                )
        else:
            self.grid_true = step_grid(self.grid_true)
            if self.show_learned and self.W_lr is not None:
                self.grid_learned = step_grid_learned(self.grid_learned, self.W_lr)

    def _status_line(self):
        # show basic state distribution + speed + mode
        def dist(g):
            c0 = int((g == 0).sum())
            c1 = int((g == 1).sum())
            c2 = int((g == 2).sum())
            tot = g.size
            return (c0 / tot, c1 / tot, c2 / tot)

        p0, p1, p2 = dist(self.grid_true)
        mode_vis = "compare" if (self.show_learned and self.compare) else "single"
        run = "paused" if self.paused else "running"
        return f"{run} | vis_mode={mode_vis} | ca={self.mode} | delay={self.delay*1000:.0f}ms | true dist: 0={p0:.2f} 1={p1:.2f} 2={p2:.2f}"

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "n":
            self.step_once = True
        elif event.key == "r":
            self._randomize()
        elif event.key == "t":
            if self.show_learned:
                self.compare = not self.compare
                self._setup_axes()
        elif event.key in ("+", "="):
            self.delay = max(0.0, self.delay * 0.8)
        elif event.key in ("-", "_"):
            self.delay = min(1.0, self.delay / 0.8)
        elif event.key == "escape":
            plt.close(self.fig)

    def run(self):
        plt.show(block=False)

        while plt.fignum_exists(self.fig.number):
            do_step = (not self.paused) or self.step_once
            self.step_once = False

            if do_step:
                self._step()

            # update images
            self.im1.set_data(self.grid_true)
            if self.im2 is not None:
                self.im2.set_data(self.grid_learned)

            self.text.set_text(self._status_line())
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            if self.delay > 0:
                time.sleep(self.delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["levin", "motif"], default="motif", help="CA rule to visualize")
    ap.add_argument("--size", type=int, default=160, help="Grid side length (square)")
    ap.add_argument("--train_samples", type=int, default=200, help="Training samples for learned rule")
    ap.add_argument("--no_learned", action="store_true", help="Hide learned CA overlay")
    args = ap.parse_args()

    vis = CAVisualizer(
        H=args.size,
        W=args.size,
        seed=0,
        train_samples=args.train_samples,
        show_learned=not args.no_learned,
        mode=args.mode,
    )
    vis.run()


if __name__ == "__main__":
    main()
