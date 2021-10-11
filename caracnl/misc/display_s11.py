import numpy as np
import matplotlib.pylab as plt
from numpy import pi
from typing import List
import warnings


def _add_to_axes(frequency: np.ndarray, s11: np.ndarray, line_style: str, axes: List[plt.Axes], **kwargs) -> None:
    axes[0].plot(frequency, 10 * np.log10(np.absolute(s11)), line_style, **kwargs)
    axes[1].plot(frequency, np.angle(s11), line_style, **kwargs)
    axes[2].plot(np.real(s11), np.imag(s11), line_style, **kwargs)


def display_s11(P_r, omega, s11, s11_model=None, title="", display_marker=False, labels=None):
    fig, axes = plt.subplots(1, 3, dpi=300,
                             figsize=(16 / 2.54, 6 / 2.54), )
    if title != "":
        fig.suptitle(title)

    f = omega / 2 / pi / 1e6
    s11 = np.array(s11)
    if s11.ndim == 1:
        s11 = np.array([s11])

    for i, (s11_i, P_r_i, c_ind) in enumerate(zip(s11, P_r, np.linspace(0, 0.8, len(s11)))):
        color = plt.get_cmap('plasma')(c_ind)
        kwargs = {'marker': "."} if display_marker else {}
        kwargs['color'] = color

        _add_to_axes(f, s11_i, "-", axes, **kwargs)
        if s11_model is not None:  # Not empty
            _add_to_axes(f, s11_model[i], '-.', axes, **kwargs)

    axes[0].set_xlabel("Frequency [MHz]")
    axes[0].set_ylabel(r"Norm $s_{11}$ [dB]")
    axes[0].grid(which="both", c="#d2e4fa")
    axes[0].tick_params(axis="both", labelsize=8)

    if labels is None:
        labels = []
        for i, P_r_i in enumerate(P_r):
            labels.append(f"{10 * np.log10(P_r_i):.1f} dBm" if i == 0 or (i == len(P_r) - 1) else '_')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axes[0].legend(labels,
                       loc='lower center',
                       fontsize=8,
                       framealpha=0.5,
                       borderpad=0.2,
                       labelspacing=0.25,
                       handlelength=1.0,
                       handletextpad=0.4)

    axes[1].set_xlabel("Frequency [MHz]")
    axes[1].set_ylabel(r"Phase $s_{11}$")
    axes[1].set_yticks([-pi, -pi / 2, 0, pi / 2, pi])
    axes[1].set_yticklabels(['-π', '-π/2', '0', 'π/2', "π"])
    axes[1].tick_params(axis="both", labelsize=8)
    axes[1].grid(c="#d2e4fa")

    x = np.linspace(0, 2 * pi, 101)
    axes[2].plot(0.5 + 0.5 * np.cos(x), 0.5 * np.sin(x), "--", color="k", linewidth=1, alpha=0.5)

    # Smith chart domain
    x_min, x_max = np.min(np.real(s11)), np.max(np.real(s11))
    y_min, y_max = np.min(np.imag(s11)), np.max(np.imag(s11))

    tolerance = 0.01
    x_min = min(np.floor(x_min*2+tolerance)/2, 0)
    x_max = max(np.ceil(x_max*2-tolerance)/2, 1)
    y_min = min(np.floor(y_min*2+tolerance)/2, -0.5)
    y_max = max(np.ceil(y_max*2-tolerance)/2, 0.5)

    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_min, y_max)

    plt.xlabel(r"Re($s_{11}$)")
    plt.ylabel(r"Im($s_{11}$)")
    axes[2].set_aspect(aspect="equal")
    axes[2].grid(c="#d2e4fa")
    axes[2].tick_params(axis="both", labelsize=8)

    plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.)
    plt.show()
