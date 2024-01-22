import numpy as np

from Billiards.physics import time_of_impact_with_wall


class Ball:
    """ A ball which is going to move in the billiard"""

    def __init__(self, pos, vel, radius=0) -> None:
        """
         Args:
            pos: Position of the ball
            vel: Velocity of the ball.
            radius: Radius pf the ball. Default: 0
           """
        self.pos = np.asarray(pos)
        self.velocity = np.asarray(vel)
        self.radius = radius



class InfiniteWall:
    """ An infinite wall where balls can collide only from one side. """

    def __init__(self, start_point, end_point, velocity, side, relativistic=False, restitution=1) -> None:
        """
         Args:
            start_point: A point of the wall.
            end_point: A point of the wall.
            velocity: Velocity of the wall. If relativistic is True, velocity must be a fraction of the speed of light
            side: Position of the wall: left, right, top, bottom.
            relativistic: True if wall is relativistic. Default: False
            restitution: Restitution of the ball. Default: 1.
                    elastic collision -> restitution = 1
                    inelastic collision -> restitution <= 0
           """
        self.start_point = np.asarray(start_point)
        self.end_point = np.asarray(end_point)
        self.velocity = np.asarray(velocity)
        self.side = side
        self.relativistic = relativistic
        self.restitution = restitution

        if self.relativistic:
            if np.linalg.norm(self.velocity) > 1:
                raise ValueError("Velocity must be less than 1")

        dx, dy = self.end_point - self.start_point
        if dx == 0 and dy == 0:
            raise ValueError(f"this is not a line.{self.start_point}, {self.end_point}, {side}")

        if side == "right" or side == "bottom":
            self._normal = -np.asarray([-dy, dx]) / np.linalg.norm([-dy, dx])
        else:
            self._normal = np.asarray([-dy, dx]) / np.linalg.norm([-dy, dx])

    def update(self, vel):
        """ Calculate the velocity of a ball after colliding with the wall.

        Args:
            vel: Velocity of the ball

        Returns:
            New velocity of the ball after impact

        """
        if self.relativistic:
            # Velocity in a relativistic billiard
            return self.relativistic_velocity(vel)
        else:
            # Velocity in a classical billiard
            return vel - (1 + self.restitution) * self._normal * np.dot((vel - self.velocity), self._normal)

    def relativistic_velocity(self, vel):
        """ Calculate the velocity of a ball after colliding with the wall when relativistic mode is enabled

        Args:
            vel: Velocity of the ball

        Returns:
            tuple: velocity of the ball after impact
        """

        if self.side == "left" or self.side == "right":
            # wall_velocity = self.velocity[0]
            # denominator = 1 - 2 * wall_velocity * vel[0] + wall_velocity * wall_velocity
            # velocity_x = (-vel[0] + 2 * wall_velocity - vel[0] * wall_velocity * wall_velocity) / denominator
            # velocity_y = (1 - wall_velocity * wall_velocity) * vel[1] / denominator
            wall_velocity = self.velocity[0]
            denominator = 1 - (1 + self.restitution) * wall_velocity * vel[0] + wall_velocity * wall_velocity * self.restitution
            velocity_x = (-vel[0] * self.restitution + (1 + self.restitution) * wall_velocity - vel[0] * wall_velocity * wall_velocity) / denominator
            velocity_y = (1 - wall_velocity * wall_velocity) * vel[1] / denominator
        else:
            wall_velocity = self.velocity[1]
            denominator = 1 - (1 + self.restitution) * wall_velocity * vel[1] + wall_velocity * wall_velocity * self.restitution
            velocity_y = (-vel[1] * self.restitution + (1 + self.restitution) * wall_velocity - vel[1] * wall_velocity * wall_velocity) / denominator
            velocity_x = (1 - wall_velocity * wall_velocity) * vel[0] / denominator

        return velocity_x, velocity_y
