import math


#####  a translation from https://github.com/eclipse-sumo/sumo/blob/main/src/microsim/cfmodels/MSCFModel_IDM.cpp
def true_acceleration(ego_speed, gap_to_pred=None, pred_speed=None, accel=0.73, decel=1.67, max_speed=30.0, min_gap=2.0, headway_time=1.5, delta=4, iterations=10, time_step=1.0):
    """
    Get true acceleration value from SUMO's IDM logic (sumo/src/microsim/cfmodels/MSCFModel_IDM.cpp).
    Calculate the acceleration of the ego vehicle using IDM logic.

    :param ego_speed: Ego vehicle speed (m/s)
    :param gap_to_pred: Gap to the preceding vehicle (m), None if no preceding vehicle
    :param pred_speed: Speed of the preceding vehicle (m/s), None if no preceding vehicle
    :param accel: Maximum acceleration (m/s^2)
    :param decel: Comfortable deceleration (m/s^2)
    :param max_speed: Desired maximum speed (m/s)
    :param min_gap: Minimum gap (m)
    :param headway_time: Desired time headway (s)
    :param delta: Acceleration exponent
    :param iterations: Number of substeps for iterative updates
    :param time_step: Simulation time step (s)
    :return: Acceleration of the ego vehicle (m/s^2)
    """
    if gap_to_pred is None or pred_speed is None:
        # Free driving (no preceding vehicle)
        acc = accel * (1 - (ego_speed / max_speed) ** delta)
        return acc

    # Ensure gap is not negative or too small
    gap_to_pred = max(gap_to_pred, 0.01)

    # Iterative calculation of speed and acceleration
    new_speed = ego_speed
    two_sqrt_accel_decel = 2 * math.sqrt(accel * decel)
    for _ in range(iterations):
        delta_v = new_speed - pred_speed  # Speed difference with the preceding vehicle
        s_star = min_gap + new_speed * headway_time + (new_speed * delta_v) / two_sqrt_accel_decel  # Desired gap
        s_star = max(s_star, 0.01)  # Avoid singularity in gap calculation

        # IDM acceleration formula
        acc = accel * (1 - (new_speed / max_speed) ** delta - (s_star / gap_to_pred) ** 2)

        # Update speed iteratively
        new_speed = max(0.0, new_speed + acc * time_step / iterations)

        # Update gap iteratively
        gap_to_pred -= max(0.0, (new_speed - pred_speed) * time_step / iterations)

    return acc