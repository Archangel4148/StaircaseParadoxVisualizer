import numpy as np
import matplotlib.pyplot as plt


def make_a_staircase(start: tuple[float, float], end: tuple[float, float], step_count: int) -> np.ndarray:
    x0, y0 = start
    x1, y1 = end

    xs = np.linspace(x0, x1, step_count + 1)
    ys = np.linspace(y0, y1, step_count + 1)

    # Repeat and interleave so it makes a staircase
    X = np.repeat(xs[1:], 2)
    Y = np.empty_like(X)
    Y[0::2] = ys[:-1]
    Y[1::2] = ys[1:]

    # Include the starting point at the beginning
    X = np.concatenate(([x0], X))
    Y = np.concatenate(([y0], Y))

    # Stack into points
    points = np.column_stack((X, Y))

    return points


def main():
    plot_range = (-0.1, 1.1)
    marker = None

    start_pt, end_pt = (0, 0), (1, 1)
    step_counts = [1, 5, 25, 125, 625]
    for count in step_counts:
        pts = make_a_staircase(start_pt, end_pt, count)
        plt.plot(pts[:, 0], pts[:, 1], marker=marker, linestyle="-")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.title("Staircase Path")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()



