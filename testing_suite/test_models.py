import caracnl
import numpy as np

# Test parameters
Q_c, Q_lin, omega_0 = 100, 200, 64e6 * 2 * np.pi
i = 10
omega = np.linspace(omega_0 * (-i / Q_c + 1), omega_0 * (+i / Q_c + 1), 2 ** 8 + 1)
P_r = np.linspace(-10, 20, 10)
P_r = 10**(P_r/10)


class TestingSuite:
    def test_linear_model(self) -> None:
        P_r = np.array([11., 12])
        lin_mod = caracnl.models.get_model(Q_c, Q_lin, omega_0)
        s11 = lin_mod.evaluate(omega, P_r)
        caracnl.display_s11(P_r, omega, s11, title="test linear_model")

    def test_mono_nonlinear_model(self) -> None:
        params = caracnl.models.NonlinearModelParameters(15, 14, 1.)

        nonlinear_model = caracnl.models.get_model(Q_c, Q_lin, omega_0, params)

        s11 = nonlinear_model.evaluate(omega, P_r)

        caracnl.display_s11(P_r, omega, s11, title="test mono_nonlinear_model")

    def test_multi_nonlinear_model(self) -> None:
        P_r = np.logspace(-1, 1, 5)
        params = [caracnl.models.NonlinearModelParameters(3, 5, 2.),
                  caracnl.models.NonlinearModelParameters(3, 2, 0.5)]

        nonlinear_model = caracnl.models.get_model(Q_c, Q_lin, omega_0, params)

        s11 = nonlinear_model.evaluate(omega, P_r)
        caracnl.display_s11(P_r, omega, s11, title="test multi_nonlinear_model", display_marker=False)

    def test_display_s11(self) -> None:
        params = caracnl.models.NonlinearModelParameters(3, 5, 1.)
        nonlinear_model = caracnl.models.get_model(Q_c, Q_lin, omega_0, params)
        s11 = nonlinear_model.evaluate(omega, P_r)

        caracnl.display_s11(P_r, omega, s11, s11*1.1,  title="test display_s11", display_marker=False)

    def test_import(self) -> None:
        pass


if __name__ == '__main__':
    testing_suite = TestingSuite()

    # testing_suite.test_linear_model()
    # testing_suite.test_mono_nonlinear_model()
    # testing_suite.test_multi_nonlinear_model()
    testing_suite.test_display_s11()
