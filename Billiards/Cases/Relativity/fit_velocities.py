from Billiards.physics import calc_next_obstacle
from Billiards.obstacles import InfiniteWall, Ball
from Billiards.plotter import Plotter

import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import stats


class Simulation1D:
    """
      Place where the simulation is done
    """

    def __init__(self, wall_velocities, Relativistic_mode=False, restitution=1):
        """
         Args:
            wall_velocities: List of wall velocities.
            Relativistic_mode: True or False. Default False.
            restitution: Restitution of the ball. Default: 1.
                    elastic collision -> restitution = 1
                    inelastic collision -> 0 <= restitution < 1
       """

        self.INF = float("inf")
        self.Relativistic_mode = Relativistic_mode
        self.restitution = restitution

        # Velocities of walls
        self.top_wall_velocity = np.asarray(wall_velocities[0])
        self.bottom_wall_velocity = np.asarray(wall_velocities[1])
        self.left_wall_velocity = np.asarray(wall_velocities[2])
        self.right_wall_velocity = np.asarray(wall_velocities[3])

        # Positions of the walls
        self.top_wall_position = np.array([[0, 1000], [2, 1000]], dtype=np.float64)
        self.bottom_wall_position = np.array([[0, 1], [2, 1]], dtype=np.float64)
        self.left_wall_position = np.array([[1, 0], [1, 1]])
        self.right_wall_position = np.array([[1000, 0], [1000, 1]])

        self.top_wall_positions = []
        self.bottom_wall_positions = []
        self.left_wall_positions = []
        self.right_wall_positions = []

        self.all_ball_velocities = []
        self.initials_velocities = []
        self.a_values = []
        self.b_values = []
        self.c_values = []
        self.r_squared = []

    def evolve(self, num_of_iterations, nmax):
        """
        Args:
             num_of_iterations: Number of collision/iterations.
             nmax: Maximum number of particles.
        """
        list_velocities_modulus = []
        velocities_of_balls = np.linspace(0.001, 0.9, nmax)
        self.all_ball_velocities = velocities_of_balls
        ball_velocities_sum = np.zeros(num_of_iterations + 1)
        for j in tqdm(range(nmax)):
            # The code only support positions in the positive XY plane
            self.top_wall_position = np.array([[0, 1000], [2, 1000]], dtype=np.float64)
            self.bottom_wall_position = np.array([[0, 1], [2, 1]], dtype=np.float64)
            self.left_wall_position = np.array([[1, 0], [1, 1]])
            self.right_wall_position = np.array([[1000, 0], [1000, 1]])

            self.right_wall_velocity = np.array([-velocities_of_balls[j], 0])

            # Creating a ball
            if self.Relativistic_mode:
                x, y = random.randint(3, 1000), 100
                # vx, vy = velocities_of_balls[j], 0  # Velocidades distintas y paredes cte
                vx, vy = 0.2, 0

            pos_ball = np.array([x, y])
            vel1 = np.asarray([vx, vy])
            ball = Ball(pos_ball, vel1)

            # Array which will store the properties of the ball for each collision
            ball_positions = [pos_ball]
            ball_velocities = [vel1]

            # Array which will store the position of walls for each collision
            self.top_wall_positions = [self.top_wall_position]
            self.bottom_wall_positions = [self.bottom_wall_position]
            self.left_wall_positions = [self.left_wall_position]
            self.right_wall_positions = [self.right_wall_position]

            # Starting simulation time
            simulation_time = [0]
            time = 0

            df = pd.DataFrame()

            # Number of collisions and starting the simulation
            i = 0
            while i < num_of_iterations:

                # If two walls are at the same position there is no billiard.
                if self.top_wall_position[0][1] <= self.bottom_wall_position[0][1] or self.left_wall_position[1][0] >= \
                        self.right_wall_position[1][0]:
                    # print("There is no more billiard. Some walls have merged.")
                    break

                # Re-creating walls with the new position
                top_wall = InfiniteWall(self.top_wall_position[0], self.top_wall_position[1], self.top_wall_velocity,
                                        side="top",
                                        relativistic=self.Relativistic_mode, restitution=self.restitution)
                bottom_wall = InfiniteWall(self.bottom_wall_position[0], self.bottom_wall_position[1],
                                           self.bottom_wall_velocity,
                                           side="bottom",
                                           relativistic=self.Relativistic_mode, restitution=self.restitution)
                left_wall = InfiniteWall(self.left_wall_position[0], self.left_wall_position[1],
                                         self.left_wall_velocity,
                                         side="left",
                                         relativistic=self.Relativistic_mode, restitution=self.restitution)
                right_wall = InfiniteWall(self.right_wall_position[0], self.right_wall_position[1],
                                          self.right_wall_velocity,
                                          side="right",
                                          relativistic=self.Relativistic_mode, restitution=self.restitution)

                obstacles = [top_wall, bottom_wall, left_wall, right_wall]

                # Time to the next collision and with the resulting wall
                times_obstacles = calc_next_obstacle(ball.pos, ball.velocity, ball.radius, obstacles)

                t = times_obstacles[0][0]

                if t == self.INF:
                    # print(f"No more collisions after {i} collisions")
                    break

                # Update properties of the ballΩ
                if times_obstacles[1][0] - times_obstacles[0][
                    0] < 1e-9:  # Time difference is to close, it collides with a corner
                    if times_obstacles[0][1].side == "top" or times_obstacles[0][1].side == "bottom":
                        new_ball_velocity_Y = times_obstacles[0][1].update(vel1)[1]
                        new_ball_velocity_X = times_obstacles[1][1].update(vel1)[0]
                    else:
                        new_ball_velocity_X = times_obstacles[0][1].update(vel1)[0]
                        new_ball_velocity_Y = times_obstacles[1][1].update(vel1)[1]
                    vel1 = (new_ball_velocity_X, new_ball_velocity_Y)
                    pos_ball = ball.pos + ball.velocity * t
                    ball = Ball(pos_ball, vel1, 0.0)
                else:
                    new_ball_velocity = times_obstacles[0][1].update(vel1)
                    pos_ball = ball.pos + ball.velocity * t
                    vel1 = new_ball_velocity
                    ball = Ball(pos_ball, vel1, 0.0)

                # Store times
                # in relativistic mode it is needed to refactor the time due to dilation
                time += t
                simulation_time.append(time)

                # Store ball's properties
                ball_positions.append(pos_ball)
                ball_velocities.append(vel1)

                # Store walls' properties
                self.top_wall_position = self.top_wall_position + self.top_wall_velocity * t
                self.bottom_wall_position = self.bottom_wall_position + self.bottom_wall_velocity * t
                self.left_wall_position = self.left_wall_position + self.left_wall_velocity * t
                self.right_wall_position = self.right_wall_position + self.right_wall_velocity * t

                # Adding walls' position to the array
                self.top_wall_positions.append(self.top_wall_position)
                self.bottom_wall_positions.append(self.bottom_wall_position)
                self.left_wall_positions.append(self.left_wall_position)
                self.right_wall_positions.append(self.right_wall_position)

                i += 1

            # Speed of the ball
            ball_velocities_modulus = np.linalg.norm(ball_velocities, axis=1)
            self.all_ball_velocities = ball_velocities_modulus
            a, b, c, r_squared = self.fit_values()
            self.a_values.append(a)
            self.b_values.append(b)
            self.c_values.append(c)
            self.r_squared.append(r_squared)
            # print(self.all_ball_velocities[j], a, b, c, r_squared)
            # if len(ball_velocities_modulus) <= num_of_iterations:
            #     add = [ball_velocities_modulus[-1]] * (num_of_iterations + 1 - len(ball_velocities_modulus))
            #     ball_velocities_modulus = np.append(ball_velocities_modulus, add)

            # ball_velocities_sum = ball_velocities_sum + ball_velocities_modulus
            # list_velocities_modulus.append(ball_velocities_modulus)

        # return ball_velocities_sum / nmax, ball_positions

        with open('/Users/borjasanchezgonzalez/Desktop/parametros__V-0.2.txt', 'w+') as fh:
            for v01, a1, b1, c1, r_squared1 in zip(velocities_of_balls, self.a_values, self.b_values, self.c_values,
                                                   self.r_squared):
                fh.write('{} {} {} {} {}\n'.format(v01, a1, b1, c1, r_squared1))

    def fit_values(self):
        n = np.arange(len(self.all_ball_velocities))
        v_gran = self.all_ball_velocities
        popt, _ = curve_fit(self.fit_function, n, v_gran)
        a, b, c = popt
        # summarize the parameter values
        residuals = v_gran - self.fit_function(n, a, b, c)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((v_gran - np.mean(v_gran)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # x_line = np.arange(min(n) - min(n) * 0.1, max(n) + max(n) * 0.07, 1)

        # calculate the output for the range
        # y_line = fit_function(x_line, a, b, c)
        return a, b, c, r_squared

    def fit_function(self, n, a, b, c):
        # a = gamma, b = beta, c = alpha
        return 1 - a * np.exp(- b * n ** c)


# -------------------------------------
Relativistic_mode = True
Coef_restitution = 1
# -------------------------------------


# -------------------------------------
# Velocities of walls
# -------------------------------------
top_wall_velocity = np.array([0, 0.0], dtype=np.float64)
bottom_wall_velocity = np.array([0, 0.0], dtype=np.float64)
left_wall_velocity = np.array([0.0, 0])
right_wall_velocity = np.array([-0.01, 0])

wall_velocities = [top_wall_velocity, bottom_wall_velocity, left_wall_velocity, right_wall_velocity]

# -------------------------------------
# SIMULATION
# -------------------------------------
billiard = Simulation1D(wall_velocities, Relativistic_mode, Coef_restitution)

# Number of collision and particles
num_of_iterations = 1000
nmax = 100

# Run simulations
billiard.evolve(num_of_iterations, nmax)
