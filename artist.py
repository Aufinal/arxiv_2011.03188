""" For plotting eigenvalues"""
import matplotlib.pyplot as plt
import numpy as np


def plot(l, spec):
    """Draws a horitontal line at each value of l.
    Then does a scatterplot of real values of spec onto imaginary values of spec."""

    r = np.sqrt(max(l))

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.scatter(spec.real, spec.imag, s=1)
    for val in l:
        ax.axvline(val, c="g", linewidth=0.5)
    circle = plt.Circle((0, 0), r, color="b", linestyle="--", linewidth=0.3, fill=False)
    ax.add_artist(circle)
    plt.savefig("test1.png")
    plt.show()
