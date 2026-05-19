from src.model import load_data, gradient_descent
from src.visualize import plot_convergence, plot_fit

filepath = 'data/points.csv'

x, y = load_data(filepath)

starting_slope = 0.0
starting_intercept = 0.0
learning_rate = 0.001
epochs = 1000

print('starting gradient descent...')

final_slope, final_intercept, loss_history = gradient_descent(
    x, y, starting_slope, starting_intercept, learning_rate, epochs
)

print(f'final slope: {final_slope:.4f}')
print(f'final intercept: {final_intercept:.4f}')
print(f'final loss: {loss_history[-1]:.4f}')

plot_convergence(loss_history)
plot_fit(x, y, final_slope, final_intercept)

print('done.')