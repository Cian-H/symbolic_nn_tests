import torch


def sech(x):
    return torch.reciprocal(torch.cosh(x))


def linear_fit(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = torch.mean(x * y) - (mean_x * mean_y)
    var_x = torch.mean(x * x) - (mean_x * mean_x)
    m = cov_xy / var_x
    c = mean_y - (m * mean_x)
    return m, c


def line(x, m, c):
    return (m * x) + c


def linear_residuals(x, y, m, c):
    return y - line(x, m, c)
