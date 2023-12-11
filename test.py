import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.add_patch(plt.Rectangle((0, 0), 1.5, 1.5,color=(0.5,0.5,0.25)))

ax.add_patch(plt.Rectangle((1.5, 0), 1.5, 1.5,color=(1,0.5,0.25)))

ax.add_patch(plt.Rectangle((3, 0), 1.5, 1.5,color=(0.5,1,0.25)))

ax.set_xlim(0, 4.5)

ax.set_yticks([])
ax.set_xticks([])

plt.show()