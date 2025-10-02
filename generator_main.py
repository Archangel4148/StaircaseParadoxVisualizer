import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def staircase_points(start, end, step_count, chunk_size=10_000):
    x0, y0 = start
    x1, y1 = end

    xs = np.linspace(x0, x1, step_count + 1)
    ys = np.linspace(y0, y1, step_count + 1)

    if ys[0] < ys[-1]:
        ys = ys[::-1]

    # Iterate in chunks
    for i in range(0, step_count, chunk_size):
        end_i = min(step_count, i + chunk_size)
        X = np.repeat(xs[i + 1:end_i + 1], 2)
        Y = np.empty_like(X)
        Y[0::2] = ys[i:end_i]
        Y[1::2] = ys[i + 1:end_i + 1]

        if i == 0:
            # include starting point only once
            X = np.concatenate(([x0], X))
            Y = np.concatenate(([y0], Y))

        yield X, Y


def setup_plot(xmin, xmax, ymin, ymax, pad=0.1):
    """Set plot limits with a relative padding."""
    xpad = (xmax - xmin) * pad
    ypad = (ymax - ymin) * pad
    plt.xlim(xmin - xpad, xmax + xpad)
    plt.ylim(ymin - ypad, ymax + ypad)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ticklabel_format(axis="x", style="sci")
    plt.title("Staircase Path")

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))

    ax.set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)


def plot_staircases(start_pt, end_pt, n_values, colors=None, save=False):
    """
    Plot staircases for each n in n_values.
    Each staircase has 2**n steps.
    """
    if colors is None:
        colors = [
            "black", "firebrick", "darkorange", "darkgreen", "teal",
            "navy", "purple", "darkmagenta", "saddlebrown", "slategray",
        ]

    x0, y0 = start_pt
    x1, y1 = end_pt
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)

    # Draw the bounding square edges
    plt.plot([xmin, xmin], [ymin, ymax], linestyle="-", color="red")
    plt.plot([xmin, xmax], [ymin, ymin], linestyle="-", color="red")

    # Plot staircases
    for i, n in enumerate(n_values):
        step_count = 2 ** n
        first_chunk = True
        for X, Y in staircase_points(start_pt, end_pt, step_count):
            plt.plot(
            X,
                Y,
                linestyle="-",
                color=colors[i % len(colors)],
                label=f"n={n}" if first_chunk else None,
            )
            first_chunk = False

    setup_plot(xmin, xmax, ymin, ymax)
    plt.legend(loc="lower left")

    if save:
        # Procedural filename
        start_str = f"{start_pt[0]}_{start_pt[1]}"
        end_str = f"{end_pt[0]}_{end_pt[1]}"
        n_str = "_".join(map(str, n_values))
        fname = f"staircase_{start_str}_to_{end_str}_n{n_str}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {fname}")
    else:
        plt.show()


def main():
    SAVE = True

    start_pt, end_pt = (0, 1), (1, 0)
    plot_staircases(start_pt, end_pt, list(range(10)), save=SAVE)
    for n in range(33):
        plot_staircases(start_pt, end_pt, [n], save=SAVE)

    if SAVE:
        # Make a zip of all PNGs in current directory
        zip_path = "all_staircases.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir("."):
                if fname.endswith(".png"):
                    zf.write(fname)
        print(f"Saved ZIP archive to {zip_path}")


if __name__ == '__main__':
    main()
