import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# import your CA step function (balanced ternary: -1,0,+1)
from ternary_life_ca import step  # from /mnt/data/ternary_life_ca.py


class TernaryLifeVisualizer:
    def __init__(self, H=160, W=160, seed=0):
        self.rng = np.random.default_rng(seed)
        self.H, self.W = H, W

        # states in {-1,0,+1}
        self.grid = self.rng.integers(-1, 2, size=(H, W), dtype=np.int8)

        # animation controls
        self.paused = False
        self.step_once = False
        self.delay = 0.02  # seconds between frames

        # colormap: -1,0,+1 -> (0,1,2) via offset
        # pick any colors you like; these are readable defaults
        self.cmap = ListedColormap(["#1f77b4", "#111111", "#ff7f0e"])

        # plotting
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.im = self.ax.imshow(
            self.grid + 1,            # map {-1,0,+1} -> {0,1,2}
            cmap=self.cmap,
            vmin=0, vmax=2,
            interpolation="nearest",
        )
        self.ax.set_title("Ternary Life CA (-1, 0, +1)")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.text = self.fig.text(0.5, 0.01, "", ha="center", va="bottom")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.tight_layout(rect=(0, 0.03, 1, 1))

    def _randomize(self):
        self.grid = self.rng.integers(-1, 2, size=(self.H, self.W), dtype=np.int8)

    def _step(self):
        self.grid = step(self.grid)

    def _status_line(self):
        tot = self.grid.size
        cneg = int((self.grid == -1).sum())
        czero = int((self.grid == 0).sum())
        cpos = int((self.grid == 1).sum())
        run = "paused" if self.paused else "running"
        return (
            f"{run} | delay={self.delay*1000:.0f}ms | "
            f"-1={cneg/tot:.2f}  0={czero/tot:.2f}  +1={cpos/tot:.2f} | "
            "keys: [space]=pause, n=step, r=randomize, +/- speed, esc=quit"
        )

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "n":
            self.step_once = True
        elif event.key == "r":
            self._randomize()
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

            self.im.set_data(self.grid + 1)
            self.text.set_text(self._status_line())
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            if self.delay > 0:
                time.sleep(self.delay)


def main():
    vis = TernaryLifeVisualizer(H=180, W=180, seed=0)
    vis.run()


if __name__ == "__main__":
    main()
