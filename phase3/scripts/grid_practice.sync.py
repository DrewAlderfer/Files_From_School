# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (learn-env)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 11)
y = np.linspace(0, 10, 11)

x_coord = []
y_coord = []
x_coord_a = []
xx, yy = np.meshgrid(x, y)
        
for row in xx:
    for point in row:
        x_coord.append(point)
for row in yy:
    for point in row:
        y_coord.append(point)
for ind, row in enumerate(xx):
    if ind % 2 == 0:
        row += .5
    for point in row:
        x_coord_a.append(point)

x_space_o = np.random.random(1000)*10
y_space_o = np.random.random(1000)*10

x_space = np.round(x_space)
y_space = np.round(y_space)
data = list(zip(x_space, y_space))

grid = list(zip(x_coord, y_coord))
data_map = []
for x1, y1 in grid:
    count = 0
    for x2, y2 in data:
        if x1 == x2 and y1 == y2:
            count += 1
    data_map.append(count)
# data_map

# %%
sns.scatterplot(x=x_coord_a, y=y_coord, s=900, hue=data_map, legend=False, palette='crest', marker="h")
# sns.scatterplot(x=x_space_o, y=y_space_o, legend=False, color="red", alpha=.3, s=5)
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_pickle("../../phase2/final/00-final-dsc-phase-2-project-v2-3/data/house_pickle.pkl")
x_col_name = "long"
y_col_name = "lat"
grid_base = 50

x_col = data[x_col_name]
y_col = data[y_col_name]

# column limits and data paramters
x_max, x_min = (x_col.max(), x_col.min())
x_span = x_max - x_min
y_max, y_min = (y_col.max(), y_col.min())
y_span = y_max - y_min

# Grid Parameters
grid_height = int(round(grid_base * y_span / x_span))

x_line = np.linspace(0, grid_base, grid_base + 1)
y_line = np.linspace(0, grid_height, grid_height + 1)

x_coord = []
y_coord = []
x_coord_a = []
xx, yy = np.meshgrid(x_line, y_line)

for row in xx:
    x_coord.extend(row)
for col in yy:
    y_coord.extend(col)
for num, row in enumerate(xx):
    if num % 2 == 0:
        x_coord_a.extend(row + 0.5)
        continue
    x_coord_a.extend(row)
    
print(f"grid width: {grid_base}, grid height: {grid_height}")

# %%

# %%
fig_size_mod = grid_height/grid_base
fig_size_mod
fig, ax = plt.subplots(figsize=(12, 12*fig_size_mod))
sns.scatterplot(x=x_coord_a, y=y_coord, s=150, legend=False, palette='crest', marker="h")
# sns.scatterplot(x=x_space_o, y=y_space_o, legend=False, color="red", alpha=.3, s=1)
plt.show()
