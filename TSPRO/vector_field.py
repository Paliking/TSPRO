import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import velocities
import constants

df = pd.read_csv('final_velos.csv')
grid_x, grid_y = np.mgrid[15:23:30j, 45:53:30j]

known_points = df[['la', 'fi']].values
values_e = df['v_e'].values
values_n = df['v_n'].values

grid_e0 = griddata(known_points, values_e, (grid_x, grid_y), method='nearest')
grid_e1 = griddata(known_points, values_e, (grid_x, grid_y), method='linear')
grid_e2 = griddata(known_points, values_e, (grid_x, grid_y), method='cubic')
grid_n0 = griddata(known_points, values_n, (grid_x, grid_y), method='nearest')
grid_n1 = griddata(known_points, values_n, (grid_x, grid_y), method='linear')
grid_n2 = griddata(known_points, values_n, (grid_x, grid_y), method='cubic')

velocities.plot_SVK(constants.BORDERS_SHP, draw=True)
Q = plt.quiver(grid_x, grid_y, grid_e1, grid_n1)
qk = plt.quiverkey(Q, 0.8, 0.8, 0.002, r'2mm/year', labelpos='E',
                       coordinates='figure', color='r')
plt.xlim(14,24)
plt.ylim(47,52.5)
plt.title("Vector field of horizontal velocities")
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.show()