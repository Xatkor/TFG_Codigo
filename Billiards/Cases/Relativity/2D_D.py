from Billiards.simulation import Simulation2D
import numpy as np
from Billiards.plotter import Plotter

""" 
Non square billiard at initial condition
"""


def main():
    # -------------------------------------
    Relativistic_mode = True
    Coef_restitution = 1
    # -------------------------------------

    # -------------------------------------
    # Distance of walls
    # -------------------------------------
    top_wall_distance = 12000
    bottom_wall_distance = 1
    left_wall_distance = 1
    right_wall_distance = 21000

    wall_distances = [top_wall_distance, bottom_wall_distance, left_wall_distance, right_wall_distance]

    # -------------------------------------
    # Velocities of walls
    # -------------------------------------
    top_wall_velocity = np.array([0, -0.02], dtype=np.float64)
    bottom_wall_velocity = np.array([0, 0.01], dtype=np.float64)
    left_wall_velocity = np.array([0.0, 0])
    right_wall_velocity = np.array([0.0, 0])

    wall_velocities = [top_wall_velocity, bottom_wall_velocity, left_wall_velocity, right_wall_velocity]

    # -------------------------------------
    # SIMULATION
    # -------------------------------------
    billiard = Simulation2D(wall_velocities, wall_distances, Relativistic_mode, Coef_restitution)

    # Number of collision and particles
    num_of_iterations = 10000
    nmax = 1000

    # Run simulations
    velocities, positions = billiard.evolve(num_of_iterations, nmax)
    # Plot velocities and path
    billiard.show_billiard(ball_velocities_average=velocities, ball_positions=[])
    # Save mean velocity
    # file_name = f"2D_D3-N{nmax}.txt"
    file_name = f"2D_DDArea.txt"
    billiard.save_results(velocities, file_name)


if __name__ == "__main__":
    main()
