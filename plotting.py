import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(traj, start=None, end=None, alpha=1.0):
    """
    Plots a single trajectory given an (n, 2) array of points.

    Parameters:
    - traj: np.ndarray of shape (n, 2), representing the trajectory.
    - start: Optional tuple (x, y) for the start point.
    - end: Optional tuple (x, y) for the end point.
    - alpha: Transparency level for the trajectory.
    """
    plt.plot(traj[:, 0], traj[:, 1], label='Trajectory', alpha=alpha * 0.2)

    if start is not None:
        plt.scatter(*start, color='green', s=100, label='Start', edgecolors='black', alpha=alpha)
    
    if end is not None:
        plt.scatter(*end, color='red', s=100, label='End', edgecolors='black', alpha=alpha)

def plot_multiple_trajectories(trajs, alpha=0.5):
    """
    Plots multiple trajectories given a (B, N, 2) array of points.

    Parameters:
    - trajs: np.ndarray of shape (B, N, 2), where B is the number of trajectories.
    - alpha: Transparency level to visualize overlapping trajectories.
    """
    trajs = jnp.asarray(trajs)
    B = trajs.shape[0]

    plt.figure(figsize=(8, 6))

    for i in range(B):
        start, end = trajs[i, 0], trajs[i, -1]
        plot_trajectory(trajs[i], start=start, end=end, alpha=alpha)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['Trajectory', 'Start', 'End'])
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_coupling(x0, x1, alpha=0.5):
    """
    Plots lines connecting pairs of points from x0 to x1.

    Parameters:
    - x0: np.ndarray of shape (B, 2), starting points.
    - x1: np.ndarray of shape (B, 2), ending points.
    - alpha: Transparency level.
    """
    B = x0.shape[0]

    plt.figure(figsize=(8, 6))

    for i in range(B):
        plt.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]], 'b-', alpha=alpha*0.2)

    plt.scatter(x0[:, 0], x0[:, 1], color='green', s=100, label='Start', edgecolors='black', alpha=alpha)
    plt.scatter(x1[:, 0], x1[:, 1], color='red', s=100, label='End', edgecolors='black', alpha=alpha)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()



def plot_mnist(images, n=10):
    fig, axs = plt.subplots(1, n, figsize=(n, 1))
    for i, ax in enumerate(axs):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.show()