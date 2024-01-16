import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class Plotter:
    def __init__(self):
        #self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.billiard_color = "#357169"
        self.path_color = "#9A1B2C"

    def create_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 7))

    def plot_billiard_rectangle(self, top_wall_positions, bottom_wall_positions, left_wall_positions,
                                right_wall_positions):
        #self.create_figure()
        for k in range(0, len(top_wall_positions)):
            rectangle = Rectangle((left_wall_positions[k][0][0], bottom_wall_positions[k][0][1]),
                                  (right_wall_positions[k][0][0] - left_wall_positions[k][0][0]),
                                  (top_wall_positions[k][0][1] - bottom_wall_positions[k][0][1]),
                                  facecolor=self.billiard_color,
                                  edgecolor='black',
                                  fill=True,
                                  lw=1,
                                  alpha=0.1)
            self.ax.add_patch(rectangle)
        self.ax.set_xlim(left_wall_positions[0][0][0], right_wall_positions[0][0][0])
        plt.axis("off")

    def plot_billiard_walls(self, top_wall_positions, bottom_wall_positions, left_wall_positions, right_wall_positions):
        self.create_figure()
        for k in range(0, len(top_wall_positions)):
            self.ax.axhline(top_wall_positions[k][0][1], color="green", alpha=(k) / len(top_wall_positions))
            self.ax.axhline(bottom_wall_positions[k][0][1], color="green", alpha=(k) / len(top_wall_positions))

            self.ax.axvline(left_wall_positions[k][0][0], color="green", alpha=(k) / len(top_wall_positions))
            self.ax.axvline(right_wall_positions[k][0][0], color="green", alpha=(k) / len(top_wall_positions))

    def plot_path(self, ball_positions):
        self.create_figure()
        ball_positions_X = [punto[0] for punto in ball_positions]
        ball_positions_Y = [punto[1] for punto in ball_positions]

        self.ax.plot(ball_positions_X, ball_positions_Y, color=self.path_color, zorder=10)
        plt.axis("off")

    def plot_velocity(self, velocity):
        self.create_figure()
        iterations = [k for k in range(len(velocity))]
        self.ax.scatter(iterations, velocity, color=self.billiard_color)
        self.ax.plot(iterations, velocity, color=self.billiard_color)

    def display(self):
        plt.show()
