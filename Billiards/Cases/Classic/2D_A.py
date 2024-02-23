from Billiards.simulation import Simulation2D
import numpy as np

""" 
Top, right wall moving.
Area increasing.
"""


def main():
    # -------------------------------------
    Relativistic_mode = False
    Coef_restitution = 1
    # -------------------------------------

    # -------------------------------------
    # Distance of walls
    # -------------------------------------
    top_wall_distance = 2001
    bottom_wall_distance = 1
    left_wall_distance = 1
    right_wall_distance = 500

    wall_distances = [top_wall_distance, bottom_wall_distance, left_wall_distance, right_wall_distance]

    # -------------------------------------
    # Velocities of walls
    # -------------------------------------
    top_wall_velocity = np.array([0, 50.0], dtype=np.float64)
    bottom_wall_velocity = np.array([0, 0.0], dtype=np.float64)
    left_wall_velocity = np.array([0.0, 0])
    right_wall_velocity = np.array([50.0, 0])

    wall_velocities = [top_wall_velocity, bottom_wall_velocity, left_wall_velocity, right_wall_velocity]

    # -------------------------------------
    # SIMULATION
    # -------------------------------------
    billiard = Simulation2D(wall_velocities, wall_distances, Relativistic_mode, Coef_restitution)

    # Number of collision and particles
    num_of_iterations = 10000
    nmax = 1

    # Run simulations
    velocities, positions = billiard.evolve(num_of_iterations, nmax)
    # Plot velocities and path
    billiard.show_billiard(ball_velocities_average=velocities, ball_positions=[])

    # Save mean velocity
    file_name = f"2D_AR-N{nmax}.txt"
    billiard.save_results(velocities, file_name)


if __name__ == "__main__":
    main()
