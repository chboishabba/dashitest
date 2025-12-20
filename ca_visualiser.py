import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Import your CA + training bits ---
# Assumes this file lives next to levin_ca_train.py
from levin_ca_train import (
    step_grid,
    make_dataset,
    features_from_grid,
    train_logreg,
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


# ---------- Visualiser ----------
class CAVisualizer:
    def __init__(self, H=128, W=128, seed=0, train_samples=200, show_learned=True):
        self.rng = np.random.default_rng(seed)
        self.H, self.W = H, W

        # state
        self.grid = self.rng.integers(0, 3, size=(H, W), dtype=np.int8)
        self.grid_true = self.grid.copy()
        self.grid_learned = self.grid.copy()

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
        # Make small dataset (same as your script, just inline)
        X, Y = make_dataset(num_samples=train_samples, H=32, W=32, seed=0)
        X_feats = np.concatenate([features_from_grid(x) for x in X], axis=0)
        Y_flat = np.concatenate([y.ravel() for y in Y], axis=0)

        t0 = time.perf_counter()
        W = train_logreg(X_feats, Y_flat, lr=5e-3, iters=300)
        t1 = time.perf_counter()
        print(f"[learn] trained log-reg in {(t1 - t0) * 1e3:.1f} ms on {train_samples} samples")
        return W

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
        self.grid = self.rng.integers(0, 3, size=(self.H, self.W), dtype=np.int8)
        self.grid_true = self.grid.copy()
        self.grid_learned = self.grid.copy()

    def _step(self):
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
        mode = "compare" if (self.show_learned and self.compare) else "single"
        run = "paused" if self.paused else "running"
        return f"{run} | mode={mode} | delay={self.delay*1000:.0f}ms | true dist: 0={p0:.2f} 1={p1:.2f} 2={p2:.2f}"

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
    vis = CAVisualizer(
        H=160,
        W=160,
        seed=0,
        train_samples=200,
        show_learned=True,  # set False if you only want the true CA
    )
    vis.run()


if __name__ == "__main__":
    main()
