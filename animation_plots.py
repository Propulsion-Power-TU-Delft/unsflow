import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()
N = 100
t = np.linspace(0, 4 * np.pi, N)
z = np.sin(t) / (1 + t**2)




def update(frame):
    x = t[0:frame]
    y = z[0:frame]
    ax.clear()
    ax.set(xlim=[np.min(t), np.max(t)], ylim=[np.min(z) - (np.max(z) - np.min(z)) / 10, np.max(z) + (np.max(z) - np.min(z)) / 10],
           xlabel=r'$x$', ylabel=r'$y$')
    ax.grid(alpha=0.2)

    ax.plot(x, y, label=r'$f(x)$')
    # ax.legend()
    if len(x>1):
        ax.scatter(x[-1], y[-1], c='red')


ani = animation.FuncAnimation(fig=fig, func=update, frames=N, interval=5)
# ani.save(filename="animation_example.gif", writer="pillow")
plt.show()
