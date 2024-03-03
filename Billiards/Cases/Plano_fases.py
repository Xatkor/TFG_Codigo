from Billiards.simulation import Simulation2D
import numpy as np
import matplotlib.pyplot as plt

""" 
Top, right wall moving.
Area decreasing.
"""


def main():
    # -------------------------------------
    Relativistic_mode = True
    Coef_restitution = 1
    # -------------------------------------

    # -------------------------------------
    # Distance of walls
    # -------------------------------------
    top_wall_distance = 1000
    bottom_wall_distance = 1
    left_wall_distance = 1
    right_wall_distance = 1000

    wall_distances = [top_wall_distance, bottom_wall_distance, left_wall_distance, right_wall_distance]

    # -------------------------------------
    # Velocities of walls
    # -------------------------------------
    top_wall_velocity = np.array([0, -0.00], dtype=np.float64)
    bottom_wall_velocity = np.array([0, 0.0], dtype=np.float64)
    left_wall_velocity = np.array([0.05, 0])
    right_wall_velocity = np.array([-0.05, 0])

    wall_velocities = [top_wall_velocity, bottom_wall_velocity, left_wall_velocity, right_wall_velocity]

    # -------------------------------------
    # SIMULATION
    # -------------------------------------
    billiard = Simulation2D(wall_velocities, wall_distances, Relativistic_mode, Coef_restitution)

    # Number of collision and particles
    num_of_iterations = 1101
    nmax = 10

    all_velocities = []
    all_positions = []
    all_angles = []
    # Run simulations
    for i in range(nmax):
        velocities, positions = billiard.evolve(num_of_iterations, nmax)
        ball_positions_X, ball_positions_Y = np.array(positions).T
        all_angles.append(np.arctan2(ball_positions_Y, ball_positions_X))
        all_velocities.append(velocities)

    n = len(all_velocities)
    for index, value in enumerate(all_velocities):
        color = "C1" if index == (n - 1) else "%.2f" % (index / n)
        plt.scatter(all_angles[index], value[:len(all_angles[index])], color=color)
    plt.show()
    # Plot velocities and path
    billiard.show_billiard(ball_velocities_average=velocities, ball_positions=[])

    # Save mean velocity
    file_name = f"2D_BB2-N{nmax}.txt"
    # billiard.save_results(velocities, file_name)


if __name__ == "__main__":
    main()
