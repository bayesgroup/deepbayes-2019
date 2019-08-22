import numpy as np
import torch

from matplotlib import pyplot as plt


def plot_1D_function(func, X, axis_bounds=([0, 1], [-6.5, 16.5])):
    """ Plot one dimensional function at given points

    Parameters
    ----------
    func : callable
        1D function to plot. It should take torch.tensor as input and
        return torch.tensor

    X : torch.tensor, shape=(n_samples, )
        Training inputs

    axis_bounds : list
        list of length 4 that defines axis bounds. Default values correspond to
        Forrester function
    """
    plt.figure(figsize=(8, 7))
    x_grid = np.linspace(axis_bounds[0][0], axis_bounds[0][1], 300)
    plt.plot(x_grid, func(torch.from_numpy(x_grid)).numpy(),
             label='Forrester function')
    plt.scatter(X.cpu().numpy(), func(X).cpu().numpy(), s=50,
                label='Initial sample')

    plt.xlabel('x', fontsize=22)
    plt.ylabel('f(x)', fontsize=22)
    plt.legend(fontsize=18, loc='upper left')
    plt.xlim(axis_bounds[0])
    plt.ylim(axis_bounds[1])


def plot_acquisition(acquisition, X, y, X_candidate):
    """
    Parameters
    ----------
    acquisition : botorch.acquisition.Acquisition

    X : torch.tensor, shape=(batch, 1, dim)
        Current design inputs

    y : .torch.tensor, shape=(n_samples, 1)
        Current design targets

    X_candidate : torch.tensor, shape=(n_candidates, 1)
        New candidate points
    """

    x_grid = torch.linspace(0, 1, 200).reshape(-1, 1, 1).to(X)
    with torch.no_grad():
        acqu = acquisition(x_grid).cpu().numpy()
        posterior = acquisition.model.posterior(x_grid)

    y_mean = posterior.mean.cpu().numpy().ravel()
    y_std = torch.sqrt(posterior.variance).numpy().ravel()
    lower = y_mean - 1.96 * y_std
    upper = y_mean + 1.96 * y_std

    if max(-acqu - min(-acqu)) > 0:
        acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))
    else:
        acqu_normalized = (-acqu - min(-acqu))

    factor = max(upper) - min(lower)

    x_grid = x_grid.cpu().numpy().ravel()

    plt.plot(X.cpu().numpy(), y.cpu().numpy(), '.r', markersize=10)

    plt.plot(x_grid, 0.2 * factor * acqu_normalized
             - abs(min(lower)) - 0.25 * factor,
             '-r', lw=2, label='Acquisition')

    plt.plot(x_grid, y_mean, '-k', lw=1, alpha=0.6)
    plt.plot(x_grid, upper, '-k', alpha=0.2)
    plt.plot(x_grid, lower, '-k', alpha=0.2)

    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    plt.fill_between(x_grid, lower.ravel(), upper.ravel(), color=color,
                     alpha=0.1)

    plt.ylim(min(lower) - 0.25 * factor,
             max(upper) + 0.05 * factor)
    plt.axvline(x=X_candidate.cpu().numpy(), color='r')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)
    plt.legend()
    plt.show()


def plot_convergence(X, y, maximize=False):
    """
    Plot convergence history: distance between consecutive x's and value of
    the best selected sample

    Parameters
    ----------
    X : torch.tensor, shape=(n_samples, dim)
        History of evaluated input values

    y : torch.tensor, shape=(n_samples,)
        History of evaluated objective values

    Returns
    -------

    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    dist = torch.norm(X[1:] - X[:-1], dim=-1).cpu().numpy()
    if maximize:
        cum_best = np.maximum.accumulate(y.cpu().numpy())
    else:
        cum_best = np.minimum.accumulate(y.cpu().numpy())

    axes[0].plot(dist, '.-', c='r',)
    axes[0].set_xlabel('Iteration', fontsize=14)
    axes[0].set_ylabel(r"$d(x_i - x_{i - 1})$", fontsize=14)
    axes[0].set_title("Distance between consecutive x's", fontsize=14)
    axes[0].grid(True)


    axes[1].plot(cum_best, '.-')
    axes[1].set_xlabel('Iteration', fontsize=14)
    axes[1].set_ylabel('Best y', fontsize=14)
    axes[1].set_title('Value of the best selected sample', fontsize=14)
    axes[1].grid(True)

    fig.tight_layout()
