import numpy as np
import matplotlib.pyplot as plt

from caracnl import diophantine_approx
from caracnl import display_s11
from caracnl.models import get_model

from lmfit import Model
from numpy import pi


def correct_s11(s11: np.ndarray, z: complex, phi: float):
    return (s11 - z) * np.exp(-1j * phi)  # Corrected data


def _complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def _fit_function_linear_model(omega, Q_c, q_lin, omega_0, phi, z_r, z_i):
    Q_lin = Q_c * q_lin
    linear_model = get_model(Q_c, Q_lin, omega_0)
    omega = omega[0:int(len(omega) / 2)]
    mod_s11 = linear_model.evaluate([1.], omega) * np.exp(1j * phi) + z_r + 1j * z_i
    return _complex_to_real(mod_s11)


def get_correction_parameters(omega: np.ndarray, s11: np.ndarray, p0=None, display=True, translation=False):
    # Model construction
    my_model = Model(_fit_function_linear_model)

    # Initial values for the model parameters
    # Gives a first approximation of the parameters value if no initial guess if provided
    if p0 is None:
        Q_c = 500
        q_lin = 10
        ind = np.argmax(np.norm(s11))
        omega_0 = omega[ind]
        z_max = s11[ind]
        z_0 = (s11[0] + s11[-1]) / 2
        phi = np.angle(z_max - z_0)
        z_r = 0 if not translation else np.real(z_0)
        z_i = 0 if not translation else np.imag(z_0)

        p0 = [Q_c, q_lin, omega_0, phi, z_r, z_i]

    parameters = my_model.make_params(Q_c=p0[0], q_lin=p0[1], omega_0=p0[2], phi=p0[3], z_r=p0[4], z_i=p0[5])
    if not translation:
        parameters['z_r'].vary = False
        parameters['z_i'].vary = False

    # Bounds on the parameters values
    parameters['Q_c'].min = 0
    parameters['q_lin'].min = 0
    parameters['omega_0'].min = 0
    parameters['phi'].min = -pi
    parameters['phi'].max = pi

    result = my_model.fit(_complex_to_real(s11), parameters, x=np.concatenate((omega.copy(), omega.copy())),)

    return result
    # t_inf_97_5 = 1.960  # coeff de student pour 3σ
    # delta_p_opt = t_inf_97_5 * np.sqrt(np.diagonal(p_cov))

    # if not translation:
    #     p_opt = np.append(p_opt, [0., 0.])
    #     delta_p_opt = np.append(delta_p_opt, [0., 0.])

    # if display:
    #     s11_mod = mod_s11_lin(omega, *tuple(p_opt[0:3]))
    #     s11_cor = correct_s11(s11, p_opt)
    #
    #     S11 = np.array([s11, s11_cor, s11_mod])  # Data avant et après correction + modèle
    #
    #     plt.figure(figsize=(16 / 2.54, 6 / 2.54), dpi=300)
    #     plt.plot(omega / 2 / pi / 1e6, np.absolute(s11_mod - s11_cor))
    #     plt.xlabel("Fréquence [MHz]", fontsize=9)
    #     plt.ylabel(r"Erreur $|s_{11}^{mod} - s_{11}^{cor}|$ ")
    #     # plt.legend(["Data", "Data corrigée", "Ajustement")
    #     plt.show()
    #
    #     display_s11(omega, S11, (f"$Q_c=${p_opt[0]:.0f} $\pm$ {delta_p_opt[0]:.0f};    " +
    #                              r"$q_{lin}=$" + f"{p_opt[1]:.3f} $\pm$ {delta_p_opt[1]:.3f};    " +
    #                              f"$\omega=${p_opt[2] / 2 / pi / 1e6:.2f} $\pm$ {delta_p_opt[2] / 2 / pi / 1e6:.2f} MHz"),
    #                 small_smith_chart=False)
    #
    # return p_opt, delta_p_opt


# def show_error_plot(X, P_s, P_VNA, res, ndPnl, ind=None):
#     epsilon = 1 / len(P_VNA) * np.sqrt(res)
#     erreur = epsilon.reshape((len(X), len(P_s)))
#
#     ind = np.argmin(epsilon) if ind is None else ind
#
#     plt.figure(figsize=(12 / 2.54, 6 / 2.54), dpi=300)
#     extent = [10 * np.log10(P_s.min()), 10 * np.log10(P_s.max()), X.min(), X.max()]
#
#     plt.imshow(erreur,
#                aspect='auto',
#                extent=extent,
#                origin='lower')
#
#     n, d, Pnl = tuple(ndPnl[ind])
#     x, y = [10 * np.log10(Pnl)], [X[np.argmin(np.absolute(X - n / d))]]
#     plt.plot(x, y, "o", c="w", alpha=0.6)
#     plt.text(x[0] - 0.2, y[0] - 0.15, f"$n/d$ = {y[0]:0.2f}\n$P_s$ = {Pnl:.2f} mW", c="w", alpha=0.6)
#     plt.xlabel(r"Puissance seuil $P_s$ [dBm]")
#     plt.ylabel(r"Exposant $n/d$ [un]")
#     plt.colorbar()
#     plt.contour(erreur,
#                 np.arange(erreur.min(), erreur.max(), erreur.min()),
#                 colors="w", extent=extent, linewidths=0.5, alpha=0.15)
#     plt.show()


# def display(P_VNA, s11_exp, s11_mod):  # affichage de l'erreur
#     plt.figure(figsize=(6 / 2.54, 6 / 2.54), dpi=200)
#     plt.semilogx(P_VNA, s11_exp, "-*")
#     plt.semilogx(P_VNA, s11_mod, "-*")
#     plt.xlabel("Puissance [mW]")
#     plt.ylabel(r"$s_{11} (\omega)$ [un]")
#     plt.tight_layout()
#     plt.legend(["Exp", "Mod"])
#     plt.show()


# def analyse_from_resonance(S11, X, P_s, P_VNA, Q_c, Q_lin, omega_0):
#     def calc_Hessian(X, P_s, res, n=1, ij=None):
#         e = res.reshape((len(X), len(P_s)))
#
#         i, j = np.unravel_index(np.argmin(e), e.shape) if ij is None else tuple(ij)
#         x1, P_s1 = X[i], P_s[j]
#         x0, P_s0 = X[i - n], P_s[j - n]
#         x2, P_s2 = X[i + n], P_s[j + n]
#
#         dx2, dx0 = x2 - x1, x1 - x0
#         dP2, dP0 = P_s2 - P_s1, P_s1 - P_s0
#
#         dd2 = np.sqrt((dx2 ** 2 + dP2 ** 2))
#         dd0 = np.sqrt((dx0 ** 2 + dP0 ** 0))
#
#         e_00, e_10, e_20 = e[i - n, j - n], e[i, j - n], e[i + n, j - n]
#         e_01, e_11, e_21 = e[i - n, j], e[i, j], e[i + n, j]
#         e_02, e_12, e_22 = e[i - n, j + n], e[i, j + n], e[i + n, j + n]
#
#         H = np.array([[((e_21 - e_11) / dx2 - (e_11 - e_01) / dx0) / (dx0 * dx2) ** 0.5,
#                        ((e_22 - e_11) / dd2 - (e_11 - e_00) / dd0) / (dd0 * dd2) ** 0.5],
#                       [((e_22 - e_11) / dd2 - (e_11 - e_00) / dd0) / (dd0 * dd2) ** 0.5,
#                        ((e_12 - e_11) / dP2 - (e_11 - e_10) / dP0) / (dP0 * dP2) ** 0.5]])
#         return H
#
#     s11_exp = np.array([np.max(np.absolute(S11_i)) for S11_i in S11])
#     ndPnl, s11_ndPnl = [], []
#     for x in X:
#         n, d = diophantine_approx(x, 50)
#         for P_s_i in P_s:
#             ndPnl_i = np.array([n, d, P_s_i])
#             s11_ndPnl_i = get_s11_vs_P(omega_0, P_VNA, ndPnl_i, omega_0, Q_c, Q_lin)
#             ndPnl.append(ndPnl_i)
#             s11_ndPnl.append(s11_ndPnl_i)
#
#     res = np.array([np.sum((s11_ndPnl_i - s11_exp) ** 2) for s11_ndPnl_i in s11_ndPnl])
#
#     return ndPnl, s11_ndPnl, res, s11_exp
