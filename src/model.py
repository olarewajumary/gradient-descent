import numpy as np
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    x = df['x'].values
    y = df['y'].values
    return x, y

def compute_loss(x, y, slope, intercept):
    predictions = slope * x + intercept
    errors = predictions - y
    loss = np.mean(errors ** 2)
    return loss

def gradient_descent(x, y, slope, intercept, lr, epochs):
    n = len(x)
    loss_history = []

    for i in range(epochs):
        predictions = slope * x + intercept
        errors = predictions - y

        slope_gradient = (2/n) * np.dot(errors, x)
        intercept_gradient = (2/n) * np.sum(errors)

        slope = slope - lr * slope_gradient
        intercept = intercept - lr * intercept_gradient

        loss = compute_loss(x, y, slope, intercept)
        loss_history.append(loss)

    return slope, intercept, loss_history