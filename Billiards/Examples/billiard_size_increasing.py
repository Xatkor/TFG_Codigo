from Billiards.physics import calc_next_obstacle
from Billiards.obstacles import InfiniteWall, Ball

import matplotlib.pyplot as plt
import numpy as np

import os


INF = float("inf")

# Velocities of walls
top_wall_velocity = np.asarray([0, 0.5], dtype=np.float64)
bottom_wall_velocity = np.asarray([0, 0], dtype=np.float64)
left_wall_velocity = np.asarray([0, 0])
right_wall_velocity = np.asarray([0.5, 0])

# Position of walls
top_wall_position = np.asarray([[0, 10], [2, 10]], dtype=np.float64)
bottom_wall_position = np.asarray([[0, 1], [2, 1]], dtype=np.float64)
left_wall_position = np.asarray([[1, 0], [1, 1]])
right_wall_position = np.asarray([[10, 0], [10, 1]])

# Creating a ball
pos_ball = np.asarray([5, 5])
vel1 = np.asarray([10, 20])
ball = Ball(pos_ball, vel1)

# Array which will storage the properties of the ball for each collision
ball_positions = [pos_ball]
ball_velocities = [vel1]

# Array which will storage the position of walls for each collision
top_wall_positions = [top_wall_position[0]]
bottom_wall_positions = [bottom_wall_position[0]]
left_wall_positions = [left_wall_position[0]]
right_wall_positions = [right_wall_position[0]]

# Plotting initial ball's position
plt.scatter(pos_ball[0], pos_ball[1], alpha=0.5, color="blue")

# Starting simulation time
simulation_time = [0]
time = 0

# Ploting initial walls' position
plt.axhline(top_wall_position[0][1], color="red")
plt.axhline(bottom_wall_position[0][1], color="red")
plt.axvline(left_wall_position[1][0], color="red")
plt.axvline(right_wall_position[1][0], color="red")

# Number of collisions and starting the simulation
num_of_iterations = 100
i = 0
while i < num_of_iterations:

    # If two walls are at the same position there is no billiard.
    if top_wall_position[0][1] == bottom_wall_position[0][1] or left_wall_position[1][0] == right_wall_position[1][0]:
        print("There is no more billiard. Some walls has fusioned.")
        break

    # Adding walls' position to teh array
    top_wall_positions.append(top_wall_position)
    bottom_wall_positions.append(bottom_wall_position)
    left_wall_positions.append(left_wall_position)
    right_wall_positions.append(right_wall_position)

    # Re-creating walls with the new position
    top_wall = InfiniteWall(top_wall_position[0], top_wall_position[1], top_wall_velocity, side="top")
    bottom_wall = InfiniteWall(bottom_wall_position[0], bottom_wall_position[1], bottom_wall_velocity, side="bottom")
    left_wall = InfiniteWall(left_wall_position[0], left_wall_position[1], left_wall_velocity, side="left")
    right_wall = InfiniteWall(right_wall_position[0], right_wall_position[1], right_wall_velocity, side="right")

    obstacles = [top_wall, bottom_wall, left_wall, right_wall]

    times_obstacles = calc_next_obstacle(ball.pos, ball.velocity, ball.radius, obstacles)

    t = times_obstacles[0][0]

    if t == INF:
        print("No more collisions")
        break

    # Update properties of the ball
    if times_obstacles[1][0] - times_obstacles[0][0] < 1e-6:  # Time difference is to close, it collides with a corner
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

    # Storage of time
    time += t
    simulation_time.append(time)

    # Storage of ball's properties
    ball_positions.append(pos_ball)
    ball_velocities.append(vel1)

    # Storage of walls' properties
    top_wall_position = top_wall_position + top_wall_velocity * t
    bottom_wall_position = bottom_wall_position + bottom_wall_velocity * t
    left_wall_position = left_wall_position + left_wall_velocity * t
    right_wall_position = right_wall_position + right_wall_velocity * t

    """ print("--"*10)
    print(f"itetarion: {i+1}")
    print(f"Collide with wall: {obstacle_collide.side}")
    print(f"Velocity of the ball after impact: {ball.velocity}")
    print(f"Ball position: {ball.pos}")
    print(f"Top-Wall position: {top_wall_position[0]}")
    print(f"Bottom-Wall position: {bottom_wall_position[0]}")
    print(f"Time to impact: {t}")
    print(f"Current time simulation: {simulation_time[i+1]}") """
    i += 1

# Separation of the components of the position vector
ball_positions_X = [punto[0] for punto in ball_positions]
ball_positions_Y = [punto[1] for punto in ball_positions]

# Plotting final walls position
# TODO: plot every wall position for every collision (the opacity should be incrementing with time step)
plt.axhline(top_wall_position[0][1], color="green")
plt.axhline(bottom_wall_position[0][1], color="green")
plt.axvline(left_wall_position[1][0], color="green")
plt.axvline(right_wall_position[1][0], color="green")

# Plot trajectories and collision points
plt.plot(ball_positions_X, ball_positions_Y, alpha=0.2, color="green")
plt.scatter(ball_positions_X, ball_positions_Y, alpha=0.5, color="red")

plt.savefig(os.path.dirname(__file__) + "/" + os.path.basename(__file__).split(".")[0] + "_path.png")

plt.show()

# Speed of the ball
ball_velocities_modulus = np.linalg.norm(ball_velocities, axis=1)

# total number of collisions
iterations = [k for k in range(len(ball_velocities_modulus))]

# Axes to plot
fig, ax = plt.subplots(3, 1)

# Graphs of ball's properties during the simulation
ax[0].plot(iterations, ball_velocities_modulus)
ax[0].title.set_text("Velocity")
ax[1].plot(iterations, ball_positions_X)
ax[1].title.set_text("X")
ax[2].plot(iterations, ball_positions_Y)
ax[2].title.set_text("Y")
plt.tight_layout()

plt.savefig(os.path.dirname(__file__) + "/" + os.path.basename(__file__).split(".")[0] + "_properties.png")

plt.show()
