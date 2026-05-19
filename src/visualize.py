import matplotlib.pyplot as plt

def plot_convergence(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='steelblue')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('output/convergence.png')
    plt.close()
    print('convergence chart saved to output/convergence.png')

def plot_fit(x, y, slope, intercept):
    predicted = slope * x + intercept
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='coral', label='actual data')
    plt.plot(x, predicted, color='steelblue', label='fitted line')
    plt.title('Gradient descent line fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/line_fit.png')
    plt.close()
    print('line fit chart saved to output/line_fit.png')