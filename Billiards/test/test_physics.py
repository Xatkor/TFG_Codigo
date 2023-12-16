from math import cos, sin, sqrt

import numpy as np
import pytest
from pytest import approx

from Billiards.physics import elastic_collision, time_of_impact, time_of_impact_with_wall

INF = float("inf")
np.seterr(divide="raise")  # use pytest.raises to catch them


def test_time_of_impact():
    # check that only relative coordinates are important
    assert time_of_impact((0, 42), (42, 0), 1, (5, 42), (41, 0), 1) == 3
    assert time_of_impact((0, 0), (0, 0), 1, (5, 0), (-1, 0), 1) == 3

    # for convenience
    def toi(p2, v2, r2, t_eps=0):
        return time_of_impact((0, 0), (0, 0), 1, p2, v2, r2, t_eps)

    # check miss
    assert toi((2, 0), (1, 0), 1) == INF
    assert toi((2, 0), (0, 1), 1) == INF

    # check head-on impact
    assert toi((3, 0), (-1, 0), 1) == 1.0
    assert toi((0, 101), (0, -33), 1) == approx(3.0)

    # check sliding past each other
    assert toi((2, 0), (0, 1), 1) == INF
    assert toi((2, 10), (0, -1), 1) == INF
    assert toi((sqrt(2), sqrt(2)), (1 - 1e-7, -1), 1) == approx(0.0, abs=1e-8)

    # check sideways collision
    # length of diagonal of a unit square is sqrt(2)
    assert toi((1, 2), (0, -1), sqrt(2) - 1) == approx(1)

    # check touching, note that this might not work so nicely with floating
    # point numbers
    assert toi((2, 0), (-1, 0), 1) == 0
    assert toi((1 + 1e-12, 1), (0, -1), sqrt(2) - 1) == approx(0.0)

    # check one side the other
    assert toi((1, 0), (-42, 0), 1) == INF
    assert toi((1, 0), (-42, 0), 10) == INF

    # check point particle
    assert toi((2, 0), (-1, 0), 0) == 1  # head-on
    assert toi((1, 0), (0, 1), 0) == INF  # slide
    # cos(60°) == 1/2 => pythagoras: sin(60°) == sqrt(1 - 1/2**2) == sqrt(3/4)
    assert toi((0.5, 1), (0, -1), 0) == approx(1 - sqrt(3 / 4))  # side

    # test touching balls and t_eps
    diag = (sqrt(2), sqrt(2))
    assert toi(diag, (-1, 0), r2=1 + 1e-5) == INF
    assert toi(diag, (-1, 0), r2=1 + 1e-5, t_eps=1e-4) == approx(0.0, abs=2e-5)
    assert toi(diag, (-1, 0), r2=1) == approx(0.0)
    assert toi((sqrt(2), sqrt(2)), (-1, 0), r2=1) == approx(0.0)

    # using t_eps to detect collision
    x, y = 2 * cos(1 / 4), 2 * sin(1 / 4)
    assert (x * x + y * y) - (1 + 1) ** 2 < 0  # rounding error => not zero
    assert toi((x, y), (-1, 0), 1) == INF  # fails to detect collision
    assert toi((x, y), (-1, 0), 1, t_eps=1e-10) == approx(0.0)


def test_elastic_collision():
    pos1, pos2 = (0, 0), (2, 0)

    def ec(vel1, vel2, mass2=1):
        v1, v2 = elastic_collision(pos1, vel1, 1, pos2, vel2, mass2)
        return (tuple(v1), tuple(v2))

    # head-on collision
    assert ec((0, 0), (-1, 0)) == ((-1, 0), (0, 0))
    assert ec((1, 0), (-1, 0)) == ((-1, 0), (1, 0))
    assert ec((1, 0), (0, 0)) == ((0, 0), (1, 0))

    # sideways collsion
    assert ec((0, 0), (-1, 1)) == ((-1, 0), (0, 1))
    assert ec((0, 0), (-0.5, 1)) == ((-0.5, 0), (0, 1))
    assert ec((0, 0), (-42, 1 / 42)) == ((-42, 0), (0, 1 / 42))

    # zero mass collision
    assert ec((-1, 0), (-20, 0), mass2=0) == ((-1, 0), (18, 0))

    # collision of two massless particles makes no sense
    with pytest.raises(FloatingPointError):
        elastic_collision(pos1, (1, 0), 0, pos2, (0, 0), 0)


def test_time_of_impact_with_wall():
    # -----------------------------------------------------

    # ------- Vertical left wall. Billiards on the right -------

    # -----------------------------------------------------

    side = "left"
    start_point = np.asarray([1, 0])
    end_point = np.asarray([1, 1])
    posW_left = start_point
    dx, dy = end_point - start_point
    _normal = np.asarray([-dy, dx])  # normal on the left
    _normal = _normal / np.linalg.norm(_normal)  # unity normal vector

    # Both moving to the right. Wall faster than ball
    vel1 = np.asarray([0.2, 0.1])
    velW = np.asarray([1, 0])
    radius = 0.2
    assert time_of_impact_with_wall((2, 1), vel1, 0.2, posW_left, velW, _normal, side) == 1.0

    # Both moving to the right. Ball faster than Wall
    vel1 = np.asarray([1, 0.1])
    velW = np.asarray([0.2, 0])
    assert time_of_impact_with_wall((2, 1), vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving to the left. Ball faster than wall
    vel1 = np.asarray([-1, 1])
    velW = np.asarray([-0.5, 0])
    # radius = 0.2
    # _normal = np.asarray([-dy, dx]) # normal on the left
    assert time_of_impact_with_wall((2, 1), vel1, radius, posW_left, velW, _normal, side) == 1.6

    # Both moving to the left. Wall faster than ball
    vel1 = np.asarray([-0.5, 1])
    velW = np.asarray([-1, 0])
    assert time_of_impact_with_wall((2, 1), vel1, radius, posW_left, velW, _normal, side) == INF

    # Head-on impact
    vel1 = np.asarray([-1, 1])
    velW = np.asarray([1, 0])
    # _normal = np.asarray([-dy, dx]) # normal on the left
    assert time_of_impact_with_wall((2, 1), vel1, 0.2, posW_left, velW, _normal, side) == 0.4

    # Both moving to the left. Equal velocities
    vel1 = np.asarray([-1, 1])
    velW = np.asarray([-1, 0])
    # _normal = np.asarray([-dy, dx]) # normal on the left
    assert time_of_impact_with_wall((2, 1), vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving to the right. Equal velocities
    vel1 = np.asarray([1, 1])
    velW = np.asarray([1, 0])
    # _normal = np.asarray([-dy, dx]) # normal on the left
    assert time_of_impact_with_wall((2, 1), vel1, 0.2, posW_left, velW, _normal, side) == INF

    # -----------------------------------------------------

    # ------- Vertical right wall. Billiards on the left -------

    # -----------------------------------------------------

    side = "right"
    start_point = np.asarray([2, 0])
    end_point = np.asarray([2, 1])
    posW_left = start_point
    dx, dy = end_point - start_point
    _normal = np.asarray([-dy, dx])  # normal on the left
    _normal = _normal / np.linalg.norm(_normal)  # unity normal vector

    # Both moving to the left. Wall faster than ball
    vel1 = np.asarray([-0.2, 0.1])
    velW = np.asarray([-1, 0])
    assert time_of_impact_with_wall((1, 1), vel1, 0.2, posW_left, velW, _normal, side) == 1.0

    # Both moving to the left. Ball faster than wall
    vel1 = np.asarray([-10, 0.1])
    velW = np.asarray([-1, 0])
    assert time_of_impact_with_wall((1, 1), vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving to the right. Ball faster than wall
    vel1 = np.asarray([10, 1])
    velW = np.asarray([3, 0])
    assert time_of_impact_with_wall((1, 1), vel1, 0.2, posW_left, velW, _normal, side) == 0.1142857142857143

    # Both moving to the right. Wall faster than ball
    vel1 = np.asarray([1, 1])
    velW = np.asarray([30, 0])
    assert time_of_impact_with_wall((1, 1), vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Head-on impact
    vel1 = np.asarray([2, 1])
    velW = np.asarray([-3, 0])
    assert time_of_impact_with_wall((1, 1), vel1, 0.2, posW_left, velW, _normal, side) == 0.16

    # Both moving the right. Equal velocities
    vel1 = np.asarray([2, 1])
    velW = np.asarray([2, 0])
    assert time_of_impact_with_wall((1, 1), vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving the LEFT. Equal velocities
    vel1 = np.asarray([-2, 1])
    velW = np.asarray([-2, 0])
    assert time_of_impact_with_wall((1, 1), vel1, 0.2, posW_left, velW, _normal, side) == INF

    # -----------------------------------------------------

    # ------- Horizontal top wall. Billiards below -------

    # -----------------------------------------------------

    side = "top"
    pos_ball = np.asarray([1, 1])
    start_point = np.asarray([0, 3])
    end_point = np.asarray([2, 3])
    posW_left = start_point
    dx, dy = end_point - start_point
    _normal = np.asarray([-dy, dx])  # normal on the left
    _normal = _normal / np.linalg.norm(_normal)  # unity normal vector

    # Both moving upwards. Wall faster than ball
    vel1 = np.asarray([0.1, 0.2])
    velW = np.asarray([0, 1])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving upwards. Ball faster than wall
    vel1 = np.asarray([1, 2])
    velW = np.asarray([0, 1])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == approx(1.8)

    # Both moving downwards. Ball faster than wall
    vel1 = np.asarray([1, -10])
    velW = np.asarray([0, -3])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving downwards. Wall faster than Ball
    vel1 = np.asarray([1, -1])
    velW = np.asarray([0, -30])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == 0.06206896551724137

    # Head-on impact
    vel1 = np.asarray([1, 2])
    velW = np.asarray([0, -3])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == 0.36

    # Both moving upwards. Equal velocities
    vel1 = np.asarray([2, 10])
    velW = np.asarray([0, 10])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving downwards. Equal velocities
    vel1 = np.asarray([-2, -6])
    velW = np.asarray([0, -6])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == INF

    # -----------------------------------------------------

    # ------- Horizontal bottom wall. Billiards above -------

    # -----------------------------------------------------

    side = "bottom"
    pos_ball = np.asarray([1, 4])
    start_point = np.asarray([0, 1])
    end_point = np.asarray([2, 1])
    posW_left = start_point
    dx, dy = end_point - start_point
    _normal = np.asarray([-dy, dx])  # normal on the left
    _normal = _normal / np.linalg.norm(_normal)  # unity normal vector

    # Both moving upwards. Wall faster than ball
    vel1 = np.asarray([0.1, 0.2])
    velW = np.asarray([0, 1])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == approx(3.5)

    # Both moving upwards. Wall faster than ball
    vel1 = np.asarray([0.1, 7])
    velW = np.asarray([0, 1])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving upwards. Equal velocities
    vel1 = np.asarray([0.1, 1])
    velW = np.asarray([0, 1])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving downwards. Wall faster than ball
    vel1 = np.asarray([0.1, -1])
    velW = np.asarray([0, -10])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Both moving downwards. Ball faster than wall
    vel1 = np.asarray([0.1, -10])
    velW = np.asarray([0, -5])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == 0.5599999999999999

    # Both moving downwards. Equal velocities
    vel1 = np.asarray([0.1, -10])
    velW = np.asarray([0, -10])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == INF

    # Head-on
    vel1 = np.asarray([0.1, -3])
    velW = np.asarray([0, 10])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == approx(0.2153846153846154)
    new_pos = pos_ball + vel1 * time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side)
    assert new_pos[0] == approx(1.021538461538462), new_pos[0]


    # test_simulations.py
    side = "bottom"
    pos_ball = np.asarray([1.5, 1.8])
    start_point = np.asarray([0, 1])
    end_point = np.asarray([2, 1])
    posW_left = start_point
    dx, dy = end_point - start_point
    _normal = np.asarray([-dy, dx])  # normal on the left
    _normal = _normal / np.linalg.norm(_normal)  # unity normal vector

    vel1 = np.asarray([0, -1])
    velW = np.asarray([0, 0])
    assert time_of_impact_with_wall(pos_ball, vel1, 0.2, posW_left, velW, _normal, side) == approx(0.6)


if __name__ == "__main__":
    pytest.main(["-v", "Billares/test/test_physics.py"])