import numpy as np
import matplotlib.pylab as plt
from numpy import pi


def display_s11(omega, s11, title="", small_smith_chart=True, display_marker=False):
    if title == "":
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=300,
                                            figsize=(16 / 2.54, 6 / 2.54), )
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=300,
                                            figsize=(16 / 2.54, 6.5 / 2.54), )
        fig.suptitle(title)

    f = omega / 2 / pi / 1e6
    s11 = np.array(s11)
    if s11.ndim == 1:
        s11 = np.array([s11])

    for s11_i in s11:
        if display_marker:
            ax1.plot(f, 10 * np.log10(np.absolute(s11_i)), "-", marker=".")
            ax2.plot(f, np.angle(s11_i), "-", marker=".")
            ax3.plot(np.real(s11_i), np.imag(s11_i), "-", marker=".")
        else:
            ax1.plot(f, 10 * np.log10(np.absolute(s11_i)))
            ax2.plot(f, np.angle(s11_i))
            ax3.plot(np.real(s11_i), np.imag(s11_i))

    ax1.set_xlabel("Frequency [MHz]")
    ax1.set_ylabel(r"Norm $s_{11}$ [dB]")
    ax1.grid(which="both", c="#d2e4fa")
    ax1.tick_params(axis="both", labelsize=8)

    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel(r"Phase $s_{11}$")
    ax2.set_yticks([-pi, -pi / 2, 0, pi / 2, pi])
    ax2.set_yticklabels(['-π', '-π/2', '0', 'π/2', "π"])
    ax2.tick_params(axis="both", labelsize=8)
    ax2.grid(c="#d2e4fa")

    x = np.linspace(0, 2 * pi, 101)
    ax3.plot(0.5 + 0.5 * np.cos(x), 0.5 * np.sin(x), "--")
    if not small_smith_chart:
        ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax3.set_ylim(-1, 1)
        ax3.set_xlim(-1, 1)
    else:
        ax3.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax3.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 1])
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_xlim(0, 1)

    plt.xlabel(r"Re($s_{11}$)")
    plt.ylabel(r"Im($s_{11}$)")
    ax3.set_aspect(aspect="equal")
    ax3.grid(c="#d2e4fa")
    ax3.tick_params(axis="both", labelsize=8)

    plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.)
    plt.show()
