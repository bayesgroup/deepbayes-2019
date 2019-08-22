import torch

from matplotlib import pyplot as plt


def plot_model(model, xlim=None, scaler_x=None, scaler_y=None):
    """
    Plot 1D GP model

    Parameters
    ----------
    model : gpytorch.models.GP

    xlim : tuple(float, float) or None

    scaler_x : sklearn.preprocessing.StandardScaler

    scaler_y : sklearn.preprocessing.StandardScaler

    Returns
    -------

    """
    X = model.train_inputs[0].cpu().numpy()
    y = model.train_targets.cpu().numpy()

    if xlim is None:
        xmin = float(X.min())
        xmax = float(X.max())
        x_range = xmax - xmin
        xlim = [xmin - 0.05 * x_range,
                xmax + 0.05 * x_range]

    model_tensor_example = list(model.parameters())[0]

    x = torch.linspace(xlim[0], xlim[1], 200).to(model_tensor_example)
    if scaler_x is not None:
        x = torch.tensor(scaler_x.transform(x.reshape(-1, 1))).squeeze()

    model.eval()
    predictive_distribution = model.predict(x)

    lower, upper = predictive_distribution.confidence_region()
    prediction = predictive_distribution.mean.cpu().numpy()

    if scaler_x is not None:
        X = scaler_x.inverse_transform(X)
        x = scaler_x.inverse_transform(x)
    else:
        x = x.numpy()

    if scaler_y is not None:
        y = scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
        lower = scaler_y.inverse_transform(lower)
        upper = scaler_y.inverse_transform(upper)
        prediction = scaler_y.inverse_transform(prediction)

    plt.scatter(X, y, marker='x', c='k')
    plt.plot(x, prediction)
    plt.fill_between(x, lower, upper, alpha=0.1)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
