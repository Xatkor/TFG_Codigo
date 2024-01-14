from Billiards.physics import calc_next_obstacle
from Billiards.obstacles import InfiniteWall, Ball

import matplotlib.pyplot as plt
import numpy as np
import random

import os

INF = float("inf")

# -------------------------------------

Relativistic_mode = False

# -------------------------------------

# Velocities of walls
top_wall_velocity = np.asarray([0, 0.0], dtype=np.float64)
bottom_wall_velocity = np.asarray([0, 0.0], dtype=np.float64)
left_wall_velocity = np.asarray([15.0, 0])
right_wall_velocity = np.asarray([0.0, 0])

ball_velocities_average = np.zeros(1001)
nmax = 1000
for j in range(nmax):
    # TODO: Simulate a number of balls with different initial conditions and calculate the mean velocity

    # The code only support positions in the positive XY plane
    # Position of walls
    top_wall_position = np.asarray([[0, 3000], [2, 3000]], dtype=np.float64)
    bottom_wall_position = np.asarray([[0, 1], [2, 1]], dtype=np.float64)
    left_wall_position = np.asarray([[100, 0], [100, 1]])
    right_wall_position = np.asarray([[3000, 0], [3000, 1]])

    # Creating a ball
    x, y = random.randint(101, 2999), random.randint(2, 2999)
    vx, vy = random.randint(0, 1000), random.randint(0, 1000)
    pos_ball = np.asarray([x, y])
    vel1 = np.asarray([vx, vy])
    ball = Ball(pos_ball, vel1)
    #print(pos_ball, vel1)

    # Array which will store the properties of the ball for each collision
    ball_positions = [pos_ball]
    ball_velocities = [vel1]

    # Array which will store the position of walls for each collision
    top_wall_positions = [top_wall_position]
    bottom_wall_positions = [bottom_wall_position]
    left_wall_positions = [left_wall_position]
    right_wall_positions = [right_wall_position]

    # Starting simulation time
    simulation_time = [0]
    time = 0

    # Number of collisions and starting the simulation
    num_of_iterations = 1000
    i = 0
    while i < num_of_iterations:

        # If two walls are at the same position there is no billiard.
        if top_wall_position[0][1] < bottom_wall_position[0][1] or left_wall_position[1][0] > right_wall_position[1][0]:
            print("There is no more billiard. Some walls has merged.")
            break

        # Re-creating walls with the new position
        top_wall = InfiniteWall(top_wall_position[0], top_wall_position[1], top_wall_velocity, side="top",
                                relativistic=Relativistic_mode)
        bottom_wall = InfiniteWall(bottom_wall_position[0], bottom_wall_position[1], bottom_wall_velocity, side="bottom",
                                   relativistic=Relativistic_mode)
        left_wall = InfiniteWall(left_wall_position[0], left_wall_position[1], left_wall_velocity, side="left", relativistic=Relativistic_mode)
        right_wall = InfiniteWall(right_wall_position[0], right_wall_position[1], right_wall_velocity, side="right", relativistic=Relativistic_mode)

        obstacles = [top_wall, bottom_wall, left_wall, right_wall]

        # Time to the next collision and with the resulting wall
        times_obstacles = calc_next_obstacle(ball.pos, ball.velocity, ball.radius, obstacles)

        t = times_obstacles[0][0]

        if t == INF:
            print(f"No more collisions after {i} collisions")
            break

        # Update properties of the ball
        if times_obstacles[1][0] - times_obstacles[0][0] < 1e-9:  # Time difference is to close, it collides with a corner
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
        top_wall_position = top_wall_position + top_wall_velocity * t
        bottom_wall_position = bottom_wall_position + bottom_wall_velocity * t
        left_wall_position = left_wall_position + left_wall_velocity * t
        right_wall_position = right_wall_position + right_wall_velocity * t

        # Adding walls' position to the array
        top_wall_positions.append(top_wall_position)
        bottom_wall_positions.append(bottom_wall_position)
        left_wall_positions.append(left_wall_position)
        right_wall_positions.append(right_wall_position)

        i += 1

    # Speed of the ball
    ball_velocities_modulus = np.linalg.norm(ball_velocities, axis=1)
    if len(ball_velocities_modulus) == num_of_iterations + 1:
        for _ in range(num_of_iterations + 1 - len(ball_velocities_modulus)):
            ball_velocities_modulus.append(ball_velocities_modulus[-1])

    ball_velocities_average = ball_velocities_average + ball_velocities_modulus

# Separation of the components of the position vector
#ball_positions_X = [punto[0] for punto in ball_positions]
#ball_positions_Y = [punto[1] for punto in ball_positions]



# total number of collisions
iterations = [k for k in range(len(ball_velocities_modulus))]

# Graphs of ball's properties during the simulation
plt.plot(iterations, ball_velocities_average/nmax)
plt.xlabel('Iterations')
plt.ylabel('Velocity')
plt.show()