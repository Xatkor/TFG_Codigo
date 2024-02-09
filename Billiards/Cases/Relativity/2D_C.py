from Billiards.simulation import Simulation2D
import numpy as np


""" 
All moving.
"""

# -------------------------------------
Relativistic_mode = True
Coef_restitution = 1
# -------------------------------------


# -------------------------------------
# Velocities of walls
# -------------------------------------
top_wall_velocity = np.array([0, 0.015], dtype=np.float64)
bottom_wall_velocity = np.array([0, 0.010], dtype=np.float64)
left_wall_velocity = np.array([0.015, 0])
right_wall_velocity = np.array([0.01, 0])

wall_velocities = [top_wall_velocity, bottom_wall_velocity, left_wall_velocity, right_wall_velocity]


# -------------------------------------
# SIMULATION
# -------------------------------------
billiard = Simulation2D(wall_velocities, Relativistic_mode, Coef_restitution)

# Number of collision and particles
num_of_iterations = 5000
nmax = 1000

# Run simulations
velocities, positions = billiard.evolve(num_of_iterations, nmax)
# Plot velocities and path
billiard.show_billiard(ball_velocities_average=velocities, ball_positions=[])

# Save mean velocity
file_name = f"2D_C3-N{nmax}.txt"
billiard.save_results(velocities, file_name)
