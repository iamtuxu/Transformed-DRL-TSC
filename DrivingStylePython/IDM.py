import numpy as np


def idm(v, v_l, d, v0, d_min, a, b, T):
    """
    Intelligent Driver Model (IDM) function.

    Parameters:
    v : float
        Current speed of the vehicle.
    v0 : float
        Desired speed of the vehicle.
    vl : float
        speed of the leader vehicle.
    d : float
        Gap to the vehicle in front.
    a : float
        Maximum acceleration.
    b : float
        Comfortable deceleration.
    T : float
        Desired time headway.
    delta : float
        Acceleration exponent.

    Returns:
    float
        Calculated acceleration based on IDM.
    """

    d_star = d_min + max(0, v * T - (v * (v_l - v)) / (2 * np.sqrt(a * b)))
    acc = max(a * (1 - (v / v0)**4 - (d_star / d)**2), -4)
    return acc


# # Example usage
# v = 20  # Current speed (m/s)
# v0 = 30  # Desired speed (m/s)
# v_l = 30 # leader_speed
# d = 10  # Gap to the vehicle in front (m)
# d_min = 7
# a = 1  # Maximum acceleration (m/s^2)
# b = 4  # Comfortable deceleration (m/s^2)
# T = 1  # Desired time headway (s)
#
# # Calculate acceleration using IDM
# acc = idm(v, v0, v_l, d, d_min, a, b, T)
# print("Acceleration:", acc)

