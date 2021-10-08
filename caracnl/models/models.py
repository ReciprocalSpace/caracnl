# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:28:01 2021

@author: Aimé Labbé
"""

import numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List


@dataclass
class NonlinearModelParameters:
    """Contains the parameter values for a nonlinear mechanism"""
    n: int  # Nominator of the power law defining the non-linear mechanism
    d: int  # Denominator of the power law defining the non-linear mechanism
    P_cnl: float  # Readout power level at which the non-linear mechanism is expected to be seen


class BaseModel(ABC):
    """This interface declares the method common to all models"""

    def __init__(self, Q_c: float, Q_lin: float, omega_0: float) -> None:
        self.Q_c = Q_c
        self.Q_lin = Q_lin
        self.omega_0 = omega_0

    @abstractmethod
    def evaluate(self, omega, P_vna):
        pass


class LinearModel(BaseModel):
    """Linear model of the resonator
    
    Model where the resonator is linear, with a coupling quality factor Q_c 
    and a linear quality factor Q_lin, and resonating at frequency omega_0.
    
    Parameters
    ----------
    Q_c : float
        Coupling quality factor. 
    Q_lin : float
        Linear quality factor of the resonator.
    omega_0 : float
        Resonant frequency of the resonator
    """

    def __init__(self, Q_c, Q_lin, omega_0) -> None:
        super().__init__(Q_c, Q_lin, omega_0)

    def evaluate(self, omega: np.ndarray, P_r=np.array([1])) -> np.ndarray:
        """
        Evaluate the model at a set of frequencies defined by omega.

        Parameters
        ----------
        omega : numpy.ndarray 
            Vector of frequency where the model is evaluated.
        P_r : numpy.ndarray, optional
            Vector of readout powers where the model is evaluated. In practice,
            this parameter is not relevant for the linear model. However, it
            is kept to be consistent with the other models.

        Returns
        -------
        s11 : numpy.ndarray
            Computed values of the reflexion coefficient s11 at the frequencies
            omega.

        """
        Q_c = self.Q_c
        q_lin = self.Q_lin / self.Q_c
        omega_0 = self.omega_0
        q_t = (1 + q_lin ** -1) ** -1
        y = Q_c * (omega - omega_0) / omega_0
        s11 = q_t * 1 / (1 + 2j * q_t * y)
        S11 = np.array([s11.copy() for _ in range(len(P_r))])
        return S11


class BaseNonlinearModel(BaseModel):
    """The interface declares the method common to all non-linear models."""

    def evaluate(self, omega, P_vna):
        pass

    @staticmethod
    def get_roots(y: float, n: int, d: int, rho_r: float, q_other: float) -> np.ndarray:
        """
        Find the roots of the characteristic polynomial for the mechanism
        described by the n/d power law. The roots represents fixed points of
        an equation which allows to compute valid values of the power dissipated
        in the non-linear mechanism.

        Then, filter the roots of the characteristic polynomial to keep only physical
        solutions. Valid solutions must be positive real numbers, and be stable
        solutions of the fixed point problem. See Thomas2020 for more information.

        Parameters
        ----------
        y : float
            Realized detuning defined by Q_c*(omega-omega_0)/omega_0
        n : int
            Numerator of the power law defining the non-linear mechanism.
        d : int
            Denominator of the power law defining the non-linear mechanism.
        rho_r : float
            Normalized readout power defined by P_r/P_nl where P_r is the
            readout power (ofter P_vna) and P_nl is the activation power of a
            particular non-linear mechanism.
        q_other : float
            Normalized quality factor defined by Q_other/Q_c where Q_other
            contains the contribution of all the other dissipation mechanism
            in the coil. If there is only one non-linear mechanism, then 
            Q_other = Q_lin. If there are more than one non-linear mechanism,
            then Q_other**-1 = Q_lin**-1 + Σ_i Q_i**-1 for all the other 
            i != j mechanisms where j denotes the mechanism under study. 

        Returns
        -------
        roots : numpy.ndarray
            Roots of the characteristic polynomial.

        """

        if n == d:  # Special case n == d -> x=1
            cond = (2 * rho_r) <= ((1 + q_other ** -1) ** 2 + (2 * y) ** 2)
            roots = np.array([0 if cond else -(1 + q_other ** -1) + np.sqrt(2 * rho_r - (2 * y) ** 2)])
            return roots  # The solution is direct

        # If n!=d we need to computes the roots.
        # The specific values of the coefficients are described in Thomas2020
        coef = []

        for i in range(2 * n + d + 1):  # Length of the polynomial
            if i == 0:
                coef.append(1.)
            elif i == 2 * n:
                coef.append((1 + q_other ** -1) ** 2 + (2 * y) ** 2)
            elif i == n:
                coef.append(2 * (1 + q_other ** -1))
            elif i == n + d:
                coef.append(-2 * rho_r)
            else:
                coef.append(0.)

        roots = np.roots(coef)
        roots = np.unique(roots)

        # Filter the invalid roots
        # Root has to be positive, real, and stable
        # Stability depends on the relative value of n/d

        valid_roots = []

        for root in roots:
            x = root ** d
            cond1 = (np.absolute(
                (1 - (1 + q_other ** -1 + x ** (n / d)) * x / rho_r) * n / d) < 1)

            if not np.isreal(root) or root < 0:  # Invalid
                pass
            # case 1
            elif root > 0 and cond1:
                valid_roots.append(root)
            # Case 2
            elif root == 0 and n > d:
                valid_roots.append(root)
        valid_roots = np.array(valid_roots)
        return np.real(valid_roots)

    def get_q_nl_inv(self, omega, n, d, rho_r, q_other):
        # Computes the q_nl**-1 associated with a specific mechanism

        y = self.Q_c * (omega - self.omega_0) / self.omega_0  # Detuning

        roots = self.get_roots(y, n, d, rho_r, q_other)
        kappa = np.max(roots)
        rho = kappa ** d
        q_nl_inv = rho ** (n / d)
        return q_nl_inv


class MonoNonlinearModel(BaseNonlinearModel):
    """Non-linear model for a single non-linear mechanism
    
    Model when the resonator has a single non-linear mechanism. The coupling 
    quality factor is defined by Q_c and the superconducting resonator has a
    linear quality factor Q_lin and resonates at omega_0. The quality factor Q_nl 
    associated with the non-linear mechanism is modeled as a power law of the 
    power dissipated Pnl in the mechanism  : Q_nl = Q_c*( Pnl / P_cnl)**(n/d) 
    with P_cnl a parameter that determines the readout power level at which any 
    non-linear behaviour is seen, and n,d two positive integers. The value n/d 
    most be seen as rational approximation of a real number. This approximation
    is used to solve the model.
    
    In the case where more than one non-linear processes are involved, use the 
    MultiNonLinModel class instead.
    
    Parameters
    ----------
    Q_c : float
        Coupling quality factor. 
    Q_lin : float
        Linear quality factor of the resonator.
    omega_0 : float
        Resonant frequency of the resonator
    model_parameters : NonLinModelParameters
        List of the different parameters from the different dissipative mechanism.
        Each mechanism corresponds to an element in the list.
    """

    def __init__(self, Q_c: float, Q_lin: float, omega_0: float,
                 model_parameters: NonlinearModelParameters) -> None:
        super().__init__(Q_c, Q_lin, omega_0)
        self.model_parameters = model_parameters

    # noinspection DuplicatedCode
    def evaluate(self, omega: np.ndarray, P_r: np.ndarray) -> np.ndarray:
        """
        Evaluate the non-linear model at set of frequency omega and a set of readout
        power P_r  (ofter P_vna). The output is a 2-dimensional matrix of size
        N x M, with N the number of P_r values and M the number of frequencies
        omega.

        Parameters
        ----------
        omega : numpy.ndarray
            Vector of frequencies where the model is evaluated.
        P_r : numpy.ndarray
            Vector of readout power where the model is evaluated.

        Returns
        -------
        S11 : numpy.ndarray
            Reflexion coefficient s11 evaluated at P_r and omega.
        """
        q_lin = self.Q_lin / self.Q_c
        rho_r = P_r / self.model_parameters.P_cnl

        q_nl_inv = np.zeros((len(P_r), len(omega)), dtype=complex)
        for i, rho_r_i in enumerate(rho_r):
            for j, omega_j in enumerate(omega):
                q_nl_inv[i, j] = self.get_q_nl_inv(omega_j,
                                                   self.model_parameters.n,
                                                   self.model_parameters.d,
                                                   rho_r_i, q_lin)
        q_t = (1 + q_lin ** -1 + q_nl_inv) ** -1

        y = self.Q_c * (omega - self.omega_0) / self.omega_0
        s11 = q_t * 1 / (1 + 2j * q_t * y)

        return s11


class MultiNonlinearModel(BaseNonlinearModel):
    """Non-linear model for a multiple non-linear mechanisms
        
    For more information about the model, please refer to the documentation
    in the MonoNonLinModel class.
    
    Parameters
    ----------
    Q_c : float
        Coupling quality factor. 
    Q_lin : float
        Linear quality factor of the resonator.
    omega_0 : float
        Resonant frequency of the resonator
    model_parameters : List[NonLinModelParameters]
        List of the different parameters from the different dissipative mechanism.
        Each mechanism corresponds to an element in the list.
    """

    # noinspection PyPep8Naming
    def __init__(self, Q_c: float, Q_lin: float, omega_0: float,
                 model_parameters: List[NonlinearModelParameters]) -> None:
        super().__init__(Q_c, Q_lin, omega_0)
        self.model_parameters = model_parameters

    # noinspection DuplicatedCode
    def evaluate(self, omega, P_r):
        """
        Evaluate the non-linear model at set of frequency omega and a set of readout
        power P_r  (ofter P_vna). The output is a 2-dimensional matrix of size
        N x M, with N the number of P_r values and M the number of frequencies
        omega. 
        
        This case is more complex. In this implementation, the multi-mechanism 
        model is solved iteratively by solving individually for each mechanism 
        while the others are assume to stay constant, hidden in the Q_other. 
        If the model contains only mechanisms such that ni/di < 1 for all i, 
        then it always has a solution. At high power, with a mechanism such that 
        ni/di>1 is present, the system can have no stable solution.

        Parameters
        ----------
        omega : numpy.ndarray
            Vector of frequencies where the model is evaluated.
        P_r : numpy.ndarray
            Vector of readout power where the model is evaluated.

        Returns
        -------
        S11 : numpy.ndarray
            Reflexion coefficient s11 evaluated at P_r and omega.
        """
        q_lin = self.Q_lin / self.Q_c

        q_nl_inv = np.zeros((len(P_r), len(omega), len(self.model_parameters)))

        for P_r_i, q_nl_inv_i in zip(P_r, q_nl_inv):
            for omega_j, q_nl_inv_ij in zip(omega, q_nl_inv_i):
                # delta_q: vector change in q for each mechanism; k: counter
                delta_q, it = np.ones((len(omega),)) * np.inf, 0
                while np.all(delta_q > 1e-4) and it < 20:
                    q_0 = q_nl_inv_ij.copy()

                    for k, model_parameters_k in enumerate(self.model_parameters):
                        # q_other_j contains all the other mechanism
                        rho_r = P_r_i/model_parameters_k.P_cnl
                        q_other_j = (q_lin ** -1 + np.sum(q_nl_inv_ij) - q_nl_inv_ij[k]) ** -1
                        q_nl_inv_ij[k] = self.get_q_nl_inv(omega_j,
                                                           model_parameters_k.n,
                                                           model_parameters_k.d, rho_r, q_other_j, )

                    delta_q, it = abs(q_0 - q_nl_inv_ij), it + 1
        q_t = (1 + q_lin ** -1 + np.sum(q_nl_inv, axis=2)) ** -1
        y = self.Q_c * (omega - self.omega_0) / self.omega_0
        s11 = q_t * 1 / (1 + 2j * q_t * y)
        return s11


def get_model(Q_c: float, Q_lin: float, omega_0: float,
              model_parameters=None) -> BaseModel:
    """
    Generate a model based on the input parameters injected in the method.
    At least a coupling quality factor, Q_c, a linear quality factor for the
    coil Q_lin and a resonant frequency omega_0 must be provided. This first
    case correspond to a linear model. If at least one set of model parameters
    are provided for the nonlinear mechanism, then the method returns a nonlinear
    model.

    Parameters
    ----------
    Q_c : float
        Coupling quality factor of the experiment.
    Q_lin : float
        Linear part of the quality factor of the nonlinear coil.
    omega_0 : float
        Resonant frequency of the nonlinear coil.
    model_parameters : NonlinearModelParameters |  List[NonlinearModelParameters]
        Single or list of  set(s) of parameters, where each set defines a nonlinear
        mechanism.
    Returns
    -------
    model : BaseModel
        Model of the linear of nonlinear coil.
    """

    # Case 1 : linear model
    if model_parameters is None:
        return LinearModel(Q_c, Q_lin, omega_0)

    # Case 2 : non-linear model with one mechanism
    elif isinstance(model_parameters, NonlinearModelParameters):
        return MonoNonlinearModel(Q_c, Q_lin, omega_0, model_parameters)

        # Case 3 : non-linear model with two or more mechanisms
    elif all([isinstance(element, NonlinearModelParameters) for element in model_parameters]):
        return MultiNonlinearModel(Q_c, Q_lin, omega_0, model_parameters)
