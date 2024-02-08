from Billiards.physics import calc_next_obstacle
from Billiards.obstacles import InfiniteWall, Ball
from Billiards.plotter import Plotter

import numpy as np
import random
import pandas as pd


class Simulation:
    def __init__(self, wall_velocities, Relativistic_mode=False, Plot_billiard=False, Plot_velocities=True):
        self.INF = float("inf")
        self.Relativistic_mode = Relativistic_mode
        self.plot_billiard = Plot_billiard
        self.plot_velocities = Plot_velocities

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

    def evolve(self, num_of_iterations, nmax):
        list_velocities_modulus = []

        ball_velocities_sum = np.zeros(num_of_iterations + 1)
        for j in range(nmax):
            # The code only support positions in the positive XY plane
            self.top_wall_position = np.array([[0, 1000], [2, 1000]], dtype=np.float64)
            self.bottom_wall_position = np.array([[0, 1], [2, 1]], dtype=np.float64)
            self.left_wall_position = np.array([[1, 0], [1, 1]])
            self.right_wall_position = np.array([[1000, 0], [1000, 1]])

            # Creating a ball
            if self.Relativistic_mode:
                x, y = random.randint(3, 1000), 0
                # vx, vy = 100 * random.choice([-1, 1]), 0
                vx, vy = random.uniform(-0.5, 0.5), 0
            else:
                x, y = random.randint(3, 999), 0
                # vx, vy = 100 * random.choice([-1, 1]), 0
                vx, vy = random.uniform(-1000, 1000), 0

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
                if self.top_wall_position[0][1] < self.bottom_wall_position[0][1] or self.left_wall_position[1][0] > \
                        self.right_wall_position[1][0]:
                    print("There is no more billiard. Some walls has merged.")
                    break

                # Re-creating walls with the new position
                top_wall = InfiniteWall(self.top_wall_position[0], self.top_wall_position[1], self.top_wall_velocity,
                                        side="top",
                                        relativistic=self.Relativistic_mode)
                bottom_wall = InfiniteWall(self.bottom_wall_position[0], self.bottom_wall_position[1],
                                           self.bottom_wall_velocity,
                                           side="bottom",
                                           relativistic=self.Relativistic_mode)
                left_wall = InfiniteWall(self.left_wall_position[0], self.left_wall_position[1],
                                         self.left_wall_velocity,
                                         side="left",
                                         relativistic=self.Relativistic_mode)
                right_wall = InfiniteWall(self.right_wall_position[0], self.right_wall_position[1],
                                          self.right_wall_velocity,
                                          side="right",
                                          relativistic=self.Relativistic_mode)

                obstacles = [top_wall, bottom_wall, left_wall, right_wall]

                # Time to the next collision and with the resulting wall
                times_obstacles = calc_next_obstacle(ball.pos, ball.velocity, ball.radius, obstacles)

                t = times_obstacles[0][0]

                if t == self.INF:
                    print(f"No more collisions after {i} collisions")
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
            if len(ball_velocities_modulus) <= num_of_iterations:
                add = [ball_velocities_modulus[-1]] * (num_of_iterations + 1 - len(ball_velocities_modulus))
                ball_velocities_modulus = np.append(ball_velocities_modulus, add)

            ball_velocities_sum = ball_velocities_sum + ball_velocities_modulus
            list_velocities_modulus.append(ball_velocities_modulus)

        return ball_velocities_sum / nmax

    def show_billiard(self, ball_velocities_average, ball_positions=[]):
        if self.plot_billiard:
            graph1 = Plotter()
            graph1.plot_billiard_rectangle(self.top_wall_positions, self.bottom_wall_positions,
                                           self.left_wall_positions, self.right_wall_positions)
            graph1.plot_path(ball_positions)
            graph1.display()

        if self.plot_velocities:
            graph2 = Plotter()
            graph2.plot_velocity(ball_velocities_average, self.Relativistic_mode, points=False)
            graph2.display()

    def save_results(self, ball_velocities_average, name="file.txt"):
        df = pd.DataFrame(ball_velocities_average)
        df["top_velocities"] = self.top_wall_velocity[1]
        df["bottom_velocities"] = self.bottom_wall_velocity[1]
        df["left_velocities"] = self.left_wall_velocity[0]
        df["right_velocities"] = self.right_wall_velocity[0]
        df.to_csv(name, sep="\t")
        # Save velocities as DataFrame
        # df = pd.DataFrame(list_velocities_modulus)
        # new_columns = {i: f'p{i+1}' for i in range(df.shape[1] - 1)}
        # df = df.transpose().rename(columns=new_columns)
        # df['mean'] = df.mean(axis=1)
        # df.to_csv(f"1D-N{nmax}-.txt", sep="\t")
