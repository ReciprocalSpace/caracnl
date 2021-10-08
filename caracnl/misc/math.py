import numpy as np


def diophantine_approx(x: float, Q: float):  # Find p/q with (p,q) \in Z**2 tq v-p/q < 1/Q
    """
    Find the rational approximation p/q of a real number x
    such that the error between x and v/q is less that 1/Q.
    This implementation uses the diophantine algorithm.

    Parameters
    ----------
    x: float
        Real number to approximate as a rational number.
    Q: float
        Definition of the

    Returns
    -------
    (p, q): Tuple[int]
        Tuple of two integers p and q approximating x
    """
    for q in range(1, Q + 1):
        q_v = q * x
        frac_q_v = q_v - np.floor(q_v)
        if frac_q_v <= Q**-1 or (1. - Q**-1) <= frac_q_v:
            p = round(q_v)
            return p, q
