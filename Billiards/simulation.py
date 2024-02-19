from Billiards.physics import calc_next_obstacle
from Billiards.obstacles import InfiniteWall, Ball
from Billiards.plotter import Plotter

import numpy as np
import random
import pandas as pd
from tqdm import tqdm


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
        self.top_wall_position = np.array([[0, 50000], [2, 50000]], dtype=np.float64)
        self.bottom_wall_position = np.array([[0, 1], [2, 1]], dtype=np.float64)
        self.left_wall_position = np.array([[1, 0], [1, 1]])
        self.right_wall_position = np.array([[50000, 0], [50000, 1]])

        self.top_wall_positions = []
        self.bottom_wall_positions = []
        self.left_wall_positions = []
        self.right_wall_positions = []

    def evolve(self, num_of_iterations, nmax):
        """
        Args:
             num_of_iterations: Number of collision/iterations.
             nmax: Maximum number of particles.
        """
        list_velocities_modulus = []

        ball_velocities_sum = np.zeros(num_of_iterations + 1)
        for j in tqdm(range(nmax)):
            # The code only support positions in the positive XY plane
            self.top_wall_position = np.array([[0, 50000], [2, 50000]], dtype=np.float64)
            self.bottom_wall_position = np.array([[0, 1], [2, 1]], dtype=np.float64)
            self.left_wall_position = np.array([[1, 0], [1, 1]])
            self.right_wall_position = np.array([[50000, 0], [50000, 1]])

            # Creating a ball
            if self.Relativistic_mode:
                x, y = random.randint(3, 1000), 100
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
            if len(ball_velocities_modulus) <= num_of_iterations:
                add = [ball_velocities_modulus[-1]] * (num_of_iterations + 1 - len(ball_velocities_modulus))
                ball_velocities_modulus = np.append(ball_velocities_modulus, add)

            ball_velocities_sum = ball_velocities_sum + ball_velocities_modulus
            list_velocities_modulus.append(ball_velocities_modulus)

        return ball_velocities_sum / nmax, ball_positions

    def show_billiard(self, ball_velocities_average=[], ball_positions=[]):
        """
        Args:
            ball_velocities_average: List of mean values for each collision.
            ball_positions: List of (x,y) values for the last simulated particle.
        """
        # if self.plot_billiard:
        if len(ball_positions) > 1:
            graph1 = Plotter()
            graph1.plot_billiard_rectangle(self.top_wall_positions, self.bottom_wall_positions,
                                           self.left_wall_positions, self.right_wall_positions)
            graph1.plot_path(ball_positions)
            graph1.display()

        if len(ball_velocities_average) > 1:
            graph2 = Plotter()
            graph2.plot_velocity(ball_velocities_average, self.Relativistic_mode, points=False)
            graph2.display()

    def save_results(self, ball_velocities_average, Coef_restitution=1, name="file.txt"):
        df = pd.DataFrame(ball_velocities_average)
        df["top_velocities"] = self.top_wall_velocity[1]
        df["bottom_velocities"] = self.bottom_wall_velocity[1]
        df["left_velocities"] = self.left_wall_velocity[0]
        df["right_velocities"] = self.right_wall_velocity[0]
        if Coef_restitution != 1:
            df["Coef_restitution"] = Coef_restitution
        df.to_csv(name, sep="\t")
        # Save velocities as DataFrame
        # df = pd.DataFrame(list_velocities_modulus)
        # new_columns = {i: f'p{i+1}' for i in range(df.shape[1] - 1)}
        # df = df.transpose().rename(columns=new_columns)
        # df['mean'] = df.mean(axis=1)
        # df.to_csv(f"1D-N{nmax}-.txt", sep="\t")


class Simulation2D:
    """
      Place where the simulation is done
    """

    def __init__(self, wall_velocities, wall_distances, Relativistic_mode=False, restitution=1):
        """
         Args:
            wall_velocities: List of wall velocities.
            wall_distances: List of distances of walls from origin. Top, Bottom, Left, Right
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
        self.top_wall_distance = wall_distances[0]
        self.bottom_wall_distance = wall_distances[1]
        self.left_wall_distance = wall_distances[2]
        self.right_wall_distance = wall_distances[3]

        self.top_wall_position = np.array([[0, self.top_wall_distance], [2, self.top_wall_distance]], dtype=np.float64)
        self.bottom_wall_position = np.array([[0, self.bottom_wall_distance], [2, self.bottom_wall_distance]], dtype=np.float64)
        self.left_wall_position = np.array([[self.left_wall_distance, 0], [self.left_wall_distance, 1]])
        self.right_wall_position = np.array([[self.right_wall_distance, 0], [self.right_wall_distance, 1]])

        self.top_wall_positions = []
        self.bottom_wall_positions = []
        self.left_wall_positions = []
        self.right_wall_positions = []

        self.area = []

    def evolve(self, num_of_iterations, nmax):
        """
        Args:
             num_of_iterations: Number of collision/iterations.
             nmax: Maximum number of particles.
        """
        list_velocities_modulus = []

        ball_velocities_sum = np.zeros(num_of_iterations + 1)
        for j in tqdm(range(nmax)):
            # The code only support positions in the positive XY plane
            self.top_wall_position = np.array([[0, self.top_wall_distance], [2, self.top_wall_distance]], dtype=np.float64)
            self.bottom_wall_position = np.array([[0, self.bottom_wall_distance], [2, self.bottom_wall_distance]], dtype=np.float64)
            self.left_wall_position = np.array([[self.left_wall_distance, 0], [self.left_wall_distance, 1]])
            self.right_wall_position = np.array([[self.right_wall_distance, 0], [self.right_wall_distance, 1]])

            # Creating walls
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


            # Creating a ball
            if self.Relativistic_mode:
                x, y = random.randint(self.left_wall_distance + 5, self.right_wall_distance - 5), random.randint(self.bottom_wall_distance + 5, self.top_wall_distance - 5)
                vx, vy = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
            else:
                x, y = random.randint(self.left_wall_distance + 5, self.right_wall_distance - 5), random.randint(self.bottom_wall_distance + 5, self.top_wall_distance - 5)
                vx, vy = random.uniform(-1000, 1000), random.uniform(-1000, 1000)

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
                    # print("There is no more billiard. Some walls have merged.")
                    break

                # Re-creating walls with the new position
                # top_wall = InfiniteWall(self.top_wall_position[0], self.top_wall_position[1], self.top_wall_velocity,
                #                         side="top",
                #                         relativistic=self.Relativistic_mode, restitution=self.restitution)
                # bottom_wall = InfiniteWall(self.bottom_wall_position[0], self.bottom_wall_position[1],
                #                            self.bottom_wall_velocity,
                #                            side="bottom",
                #                            relativistic=self.Relativistic_mode, restitution=self.restitution)
                # left_wall = InfiniteWall(self.left_wall_position[0], self.left_wall_position[1],
                #                          self.left_wall_velocity,
                #                          side="left",
                #                          relativistic=self.Relativistic_mode, restitution=self.restitution)
                # right_wall = InfiniteWall(self.right_wall_position[0], self.right_wall_position[1],
                #                           self.right_wall_velocity,
                #                           side="right",
                #                           relativistic=self.Relativistic_mode, restitution=self.restitution)

                # obstacles = [top_wall, bottom_wall, left_wall, right_wall]

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
                    pos_ball = ball.pos + np.multiply(ball.velocity, t)
                    ball.pos = pos_ball
                    ball.velocity = vel1
                    # ball = Ball(pos_ball, vel1, 0.0)
                else:
                    new_ball_velocity = times_obstacles[0][1].update(vel1)
                    pos_ball = ball.pos + np.multiply(ball.velocity, t)
                    vel1 = new_ball_velocity
                    ball.pos = pos_ball
                    ball.velocity = vel1
                    # ball = Ball(pos_ball, vel1, 0.0)

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

                top_wall.start_point = self.top_wall_position[0]
                top_wall.end_point = self.top_wall_position[1]
                bottom_wall.start_point = self.bottom_wall_position[0]
                bottom_wall.end_point = self.bottom_wall_position[1]
                left_wall.start_point = self.left_wall_position[0]
                left_wall.end_point = self.left_wall_position[1]
                right_wall.start_point = self.right_wall_position[0]
                right_wall.end_point = self.right_wall_position[1]

                i += 1

            # Speed of the ball
            self.area = self.calc_area()
            ball_velocities_modulus = np.linalg.norm(ball_velocities, axis=1)
            if len(ball_velocities_modulus) <= num_of_iterations:
                add = [ball_velocities_modulus[-1]] * (num_of_iterations + 1 - len(ball_velocities_modulus))
                ball_velocities_modulus = np.append(ball_velocities_modulus, add)

                if self.area[-1] < 0:
                    add = [self.area[-2]] * (num_of_iterations + 1 - len(self.area))
                else:
                    add = [self.area[-1]] * (num_of_iterations + 1 - len(self.area))
                self.area = np.append(self.area, add)

            ball_velocities_sum = ball_velocities_sum + ball_velocities_modulus
            list_velocities_modulus.append(ball_velocities_modulus)

        return ball_velocities_sum / nmax, ball_positions

    def show_billiard(self, ball_velocities_average=[], ball_positions=[]):
        """
        Args:
            ball_velocities_average: List of mean values for each collision.
            ball_positions: List of (x,y) values for the last simulated particle.
        """
        # if self.plot_billiard:
        if len(ball_positions) > 1:
            graph1 = Plotter()
            graph1.plot_billiard_rectangle(self.top_wall_positions, self.bottom_wall_positions,
                                           self.left_wall_positions, self.right_wall_positions)
            graph1.plot_path(ball_positions)
            graph1.display()

        if len(ball_velocities_average) > 1:
            graph2 = Plotter()
            graph2.plot_velocity(ball_velocities_average, self.Relativistic_mode, points=False)
            graph2.display()

    def calc_area(self):
        Ty = [self.top_wall_positions[k][0][1] for k in range(len(self.top_wall_positions))]
        By = [self.bottom_wall_positions[k][0][1] for k in range(len(self.bottom_wall_positions))]
        Lx = [self.left_wall_positions[k][0][0] for k in range(len(self.left_wall_positions))]
        Rx = [self.right_wall_positions[k][0][0] for k in range(len(self.right_wall_positions))]
        area = np.multiply(np.subtract(Ty, By), np.subtract(Rx, Lx))
        return area

    def save_results(self, ball_velocities_average, name="file.txt"):
        df = pd.DataFrame(ball_velocities_average)
        df["top_velocities"] = self.top_wall_velocity[1]
        df["bottom_velocities"] = self.bottom_wall_velocity[1]
        df["left_velocities"] = self.left_wall_velocity[0]
        df["right_velocities"] = self.right_wall_velocity[0]
        df["Coef_restitution"] = self.restitution
        # df["Vertical_distance_initial"] = self.top_wall_distance - self.bottom_wall_distance
        # df["Horizontal_distance_initial"] = self.right_wall_distance - self.left_wall_distance
        df["Area"] = self.area
        df.to_csv(name, sep="\t")
        # Save velocities as DataFrame
        # df = pd.DataFrame(list_velocities_modulus)
        # new_columns = {i: f'p{i+1}' for i in range(df.shape[1] - 1)}
        # df = df.transpose().rename(columns=new_columns)
        # df['mean'] = df.mean(axis=1)
        # df.to_csv(f"1D-N{nmax}-.txt", sep="\t")
