from Billiards.simulation import Simulation
import numpy as np

# -------------------------------------
Relativistic_mode = True
# -------------------------------------


# -------------------------------------
# Velocities of walls
# -------------------------------------
top_wall_velocity = np.array([0, 0.0], dtype=np.float64)
bottom_wall_velocity = np.array([0, 0.0], dtype=np.float64)
left_wall_velocity = np.array([0, 0])
right_wall_velocity = np.array([-0.01, 0])

wall_velocities = [top_wall_velocity, bottom_wall_velocity, left_wall_velocity, right_wall_velocity]


# -------------------------------------
# SIMULATION
# -------------------------------------
billiard = Simulation(wall_velocities, Relativistic_mode)
num_of_iterations = 1000
nmax = 5
velocities = billiard.evolve(num_of_iterations, nmax)
billiard.show_billiard(velocities)
billiard.save_results(velocities, f"1D_C_L>R-N{nmax}-vWall-Far-.txt")