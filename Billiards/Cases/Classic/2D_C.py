from Billiards.simulation import Simulation2D
import numpy as np


""" 
All walls moving. 

"""
def main():
    # -------------------------------------
    Relativistic_mode = False
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
    top_wall_velocity = np.array([0, 5.0], dtype=np.float64)
    bottom_wall_velocity = np.array([0, 1.0], dtype=np.float64)
    left_wall_velocity = np.array([9.0, 0])
    right_wall_velocity = np.array([5.0, 0])

    wall_velocities = [top_wall_velocity, bottom_wall_velocity, left_wall_velocity, right_wall_velocity]


    # -------------------------------------
    # SIMULATION
    # -------------------------------------
    billiard = Simulation2D(wall_velocities, wall_distances, Relativistic_mode, Coef_restitution)

    # Number of collision and particles
    num_of_iterations = 200
    nmax = 1000

    # Run simulations
    velocities, positions = billiard.evolve(num_of_iterations, nmax)
    # Plot velocities and path
    billiard.show_billiard(ball_velocities_average=velocities, ball_positions=[])

    # Save mean velocity
    file_name = f"2D_C-N{nmax}.txt"
    billiard.save_results(velocities, file_name)

if __name__ == "__main__":
    main()