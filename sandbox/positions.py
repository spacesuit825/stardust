import numpy as np
import matplotlib.pyplot as plt
import matplotlib


p1 = [0, 0]
p2 = [0, 0.5]
p3 = [0, 5]
p4 = [0, 0.2]

y = [0, 0, 0, 0]
x = [0, 0.5, 5, 0.2]
r = [0.1, 0.25, 0.1, 0.1]

fig, ax = plt.subplots()
ax.set_xlim(-2, 7)
ax.set_ylim(2, -2)
circles = [plt.Circle((xi,yi), radius=ri, linewidth=0) for xi, yi, ri in zip(x, y, r)]
c = matplotlib.collections.PatchCollection(circles)
ax.add_collection(c)
plt.show()