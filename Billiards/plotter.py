import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def display():
    plt.show()


class Plotter:
    def __init__(self):
        self.ax = None
        self.fig = None
        self.fig2 = None
        self.ax2 = None
        # self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.billiard_color = "#357169"
        self.path_color = "#9A1B2C"

    def plot_billiard_rectangle(self, top_wall_positions, bottom_wall_positions, left_wall_positions,
                                right_wall_positions):
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
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
        self.ax.set_ylim(min(bottom_wall_positions[:][0][1]), max(top_wall_positions[:][0][1]))

    def plot_billiard_walls(self, top_wall_positions, bottom_wall_positions, left_wall_positions, right_wall_positions):
        for k in range(0, len(top_wall_positions)):
            self.ax.axhline(top_wall_positions[k][0][1], color="green", alpha=(k) / len(top_wall_positions))
            self.ax.axhline(bottom_wall_positions[k][0][1], color="green", alpha=(k) / len(top_wall_positions))

            self.ax.axvline(left_wall_positions[k][0][0], color="green", alpha=(k) / len(top_wall_positions))
            self.ax.axvline(right_wall_positions[k][0][0], color="green", alpha=(k) / len(top_wall_positions))

    def plot_path(self, ball_positions):
        ball_positions_X, ball_positions_Y = np.array(ball_positions).T

        self.ax.plot(ball_positions_X, ball_positions_Y, color=self.path_color, zorder=10)
        plt.axis("off")

    def plot_velocity(self, velocity, relativistic=False):
        self.fig2, self.ax2 = plt.subplots(figsize=(10, 7))
        iterations = np.arange(len(velocity), dtype=int)
        self.ax2.plot(iterations, velocity, color=self.billiard_color)
        self.ax2.scatter(iterations, velocity, edgecolors=self.billiard_color, color="white", zorder=10)

        self.ax2.set_ylabel("$< v >$")
        self.ax2.set_xlabel("$n$")
        if relativistic:
            self.ax2.axhline(1, color=self.path_color, ls="--")



    def display(self):
        plt.show()
