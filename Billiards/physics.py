from math import sqrt

import numpy as np

INF = float("inf")


# SOLO SE ESTÁ USANDO LA FUNCION "time_of_impact_with_wall"

def time_of_impact(pos1: tuple, vel1: tuple, radius1: float, pos2: tuple, vel2: tuple, radius2: float,
                   t_eps=1e-10) -> float:
    """ Calculate for two moving balls the time until impact.

    Overlapping balls are not counted as a collision (because the impact already aocurred). But due to rounding errors an actual might be missed, use a larger t_eps to correct that

    Args:
        pos1: Center of the first ball.
        vel1: Velocity of the first ball.
        radius1: Radius of the first ball.
        pos2: Center of the second ball.
        vel2: Velocity of the second ball.
        radius2: Radius of the second ball.
        t_eps (optional): Correction for rounding errors, default: 1e-10.

    Return:
        Time until impact, is infinite if no impact.

    """
    pos_diff = np.subtract(pos2, pos1)
    vel_diff = np.subtract(vel2, vel1)

    pos_dot_vel = pos_diff.dot(vel_diff)  # dot profuct: pos_diff * vel_diff
    if pos_dot_vel >= 0:
        # balls are moving apart, no impact
        return INF

    pos_sqrd = pos_diff.dot(pos_diff)  # pos^2
    vel_sqrd = vel_diff.dot(vel_diff)  # vel^2
    assert vel_sqrd > 0, vel_sqrd  # note: vel_sqrd != 0

    # time of impact explain in "Notas physics"
    # time is given in the equation:
    # t^2 + b*t + c = 0
    b = pos_dot_vel / vel_sqrd  # b < 0 because pos_dot_vel < 0
    assert b < 0, b
    c = (pos_sqrd - (radius1 + radius2) ** 2) / vel_sqrd

    discriminant = b ** 2 - c
    if discriminant <= 0:  # complex solution
        # the balls miss or slide past each other
        return INF

    t1 = -b + sqrt(discriminant)
    assert t1 > 0, (t1, b, c, sqrt(discriminant))
    t2 = c / t1  # Explain in "Notas physics"

    if t2 < -t_eps:
        # if t2 is negative, then the balls overlap. this doesn't count as an impact, but if t2 is close to zero, then collision might have happened and we miss it just because rounding errors.
        return INF

    return min(t1, t2)


def elastic_collision(pos1: tuple, vel1: tuple, mass1: float, pos2: tuple, vel2: tuple, mass2: float):
    """
    Perfectly elastic collision between 2 balls.

    Args:
        pos1: Center of the first ball.
        vel1: Velocity of the first ball.
        mass1: Mass of the first ball.
        pos2: Center of the second ball.
        vel2: Velocity of the second ball.
        mass2: Mass of the second ball.

    Return:
        Two velocities after the collision:
        vel1, vel2

    """

    # TODO:
    # - Implementar la velocidad final relativista.
    # - Dejar la opcion de velocidad clasica.
    # - Si es un muro su velocidad final se mantiene (vel2_input = vel2_output)
    #   y solo cambia la velocidad de la bola, vel1.

    pos_diff = np.subtract(pos2, pos1)
    vel_diff = np.subtract(vel2, vel1)

    pos_dot_vel = pos_diff.dot(vel_diff)
    assert pos_dot_vel < 0  # colliding balls do not move apart

    dist_sqrd = pos_diff.dot(pos_diff)

    aux = 2 * (pos_dot_vel * pos_diff) / ((mass1 + mass2) * dist_sqrd)
    vel1 += mass2 * aux
    vel2 -= mass1 * aux

    return vel1, vel2


def relativistic_elastic_collision(pos1: tuple, vel1: tuple, mass1: float, pos2: tuple, vel2: tuple, mass2: float):
    """
    Perfectly elastic collision between a ball and a wall (mass_wall >> mass_ball).

    Args:
        pos1: Center of the first ball.
        vel1: Velocity of the ball as fraction of speed of light.
        mass1: Mass of the first ball.
        pos2: Center of the wall.
        vel2: Velocity of the wall as fraction of speed of light.
        mass2: Mass of the wall.

    Return:
        Two velocities after the collision:
        vel1, vel2

    """

    # TODO:
    # - Implementar la velocidad final relativista (Lo escrito ahora no es correcto).
    # - Dejar la opcion de velocidad clasica.
    # - Si es un muro su velocidad final se mantiene (vel2_input = vel2_output)
    #   y solo cambia la velocidad de la bola, vel1.

    if sqrt(np.asarray(vel1).dot(vel1)) > 1:
        raise ValueError("Velocity 1 must be less than 1.")

    if sqrt(np.asarray(vel2).dot(vel2)) > 1:
        raise ValueError("Velocity 2 must be less than 1.")

    # Ball velocity
    ux, uy = vel1

    # Wall velocity
    wx, wy = vel2

    vel2_modulus = sqrt(np.asarray(vel2).dot(vel2))
    aux = 1 - 2 * vel2_modulus * ux + wy ** 2
    new_ux = (- ux + 2 * wy - ux * wy ** 2) / aux
    new_uy = uy * (1 - wy ** 2)

    new_vel1 = (new_ux, new_uy)

    return new_vel1, vel2


def time_of_impact_with_wall(pos1, vel1, radius, posW, velW, _normal, side):
    """ Calculate when the ball collide with vertical or horizontal wall.

    Args:
        pos1: Position of the ball.
        vel1: Velocity of the ball.
        radius: Radius of the ball.
        posW: Position of the wall.
        velW: Velocity of the wall.
        normal: Normal vector of the wall.

    Returns:
        Time of the impact, it is infinite if no impact.

    """

    if _normal[0] != 0:
        # Vertical wall
        if vel1[0] == velW[0]:
            # Same velocity, never collide
            return INF
        if side == "left":
            t = (radius + posW[0] - pos1[0]) / (vel1[0] - velW[0])
        else:
            t = (-radius + posW[0] - pos1[0]) / (vel1[0] - velW[0])
    else:
        # Horizontal wall
        if vel1[1] == velW[1]:
            # Same velocity, never collide
            return INF
        if side == "top":
            t = (-radius + posW[1] - pos1[1]) / (vel1[1] - velW[1])
        else:
            t = (radius + posW[1] - pos1[1]) / (vel1[1] - velW[1])

    if t < 1e-6:
        # they don't collide
        return INF
    else:
        return t
