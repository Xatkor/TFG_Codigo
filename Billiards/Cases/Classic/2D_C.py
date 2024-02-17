from Billiards.simulation import Simulation2D
import numpy as np


""" 
All walls moving. 

"""

# -------------------------------------
Relativistic_mode = False
Coef_restitution = 1
# -------------------------------------


# -------------------------------------
# Velocities of walls
# -------------------------------------
top_wall_velocity = np.array([0, 1.0], dtype=np.float64)
bottom_wall_velocity = np.array([0, 3.0], dtype=np.float64)
left_wall_velocity = np.array([5.0, 0])
right_wall_velocity = np.array([-8.0, 0])

wall_velocities = [top_wall_velocity, bottom_wall_velocity, left_wall_velocity, right_wall_velocity]


# -------------------------------------
# SIMULATION
# -------------------------------------
billiard = Simulation2D(wall_velocities, Relativistic_mode, Coef_restitution)

# Number of collision and particles
num_of_iterations = 10000
nmax = 1000

# Run simulations
velocities, positions = billiard.evolve(num_of_iterations, nmax)
# Plot velocities and path
billiard.show_billiard(ball_velocities_average=velocities, ball_positions=[])

# Save mean velocity
file_name = f"2D_C_X-N{nmax}.txt"
billiard.save_results(velocities, file_name)
