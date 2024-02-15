import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FormatStrFormatter)
import numpy as np
from scipy.optimize import curve_fit

def fit_function(x, m, n, d):
    return m * x**2 + n * x + d

def fit_values(x, y):
    popt, _ = curve_fit(fit_function, x, y)
    a, b, c = popt
    # summarize the parameter values
    residuals = y - fit_function(x, a, b, c)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # x_line = np.arange(min(x) - min(x) * 0.1, max(x) + max(x) * 0.07, 1)
    x_line = np.linspace(min(x), max(x), 100)
    # calculate the output for the range
    y_line = fit_function(x_line, a, b, c)
    return x_line, y_line, r_squared, a, b, c

with open(f"/Users/borjasanchezgonzalez/Desktop/parametros__V-0.01.txt") as textfile:
    lines1 = [line.split() for line in textfile]

aux1 = np.array(lines1, dtype=float)

v0 = np.array(aux1[:-1, 0], dtype=np.float64)
a = np.array(aux1[:-1, 1], dtype=np.float64)
b = np.array(aux1[:-1, 2], dtype=np.float64)
c = np.array(aux1[:-1, 3], dtype=np.float64)

color_verde = "#2f9a1f" # Verde
color_azul2 = "#2c7fb8" # Azul
color_azul = "#3768be" # Azul
color_rojo = "#ac2929" # Rojo

fig2, ax2 = plt.subplots(3, 1)

x1, y1, r1, m ,n ,d = fit_values(v0, a)
ax2[0].plot(v0, a, color=color_azul2)
ax2[0].plot(x1, y1, color="black")
ax2[0].set_title(r"$\gamma$" + f" :  a = {m:.02f}  b = {n:.02f}  c = {d:.02f}  $R^2$ = {r1:.05f}")

x2, y2, r2, m ,n ,d= fit_values(v0, b)
ax2[1].plot(v0, b, color=color_azul2)
ax2[1].plot(x2, y2, color="black")
ax2[1].set_title(r"$\beta$" + f" :  a = {m:.02f}  b = {n:.02f}  c = {d:.02f}  $R^2$ = {r1:.05f}")

x3, y3, r3, m ,n ,d = fit_values(v0, c)
ax2[2].plot(v0, c, color=color_azul2)
ax2[2].plot(x3, y3, color="black")
ax2[2].set_title(r"$\alpha$" + f" :  a = {m:.02f}  b = {n:.02f}  c = {d:.02f}  $R^2$ = {r1:.05f}")

ax2[0].legend()
ax2[1].legend()
ax2[2].legend()

ax2[0].set_xlabel("$v_0$")
ax2[1].set_xlabel("$v_0$")
ax2[2].set_xlabel("$v_0$")

plt.tight_layout()

ax2[0].xaxis.set_minor_locator(AutoMinorLocator())
ax2[0].yaxis.set_minor_locator(AutoMinorLocator())
ax2[1].xaxis.set_minor_locator(AutoMinorLocator())
ax2[1].yaxis.set_minor_locator(AutoMinorLocator())
ax2[2].xaxis.set_minor_locator(AutoMinorLocator())
ax2[2].yaxis.set_minor_locator(AutoMinorLocator())
ax2[0].grid(alpha=0.2, color="gray")
ax2[1].grid(alpha=0.2, color="gray")
ax2[2].grid(alpha=0.2, color="gray")


plt.show()