import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return np.exp(x) / (1 + np.exp(x))

# Generate x values from -10 to 10
x_values = np.linspace(-10, 10, 400)

# Calculate y values using the defined function
y_values = f(x_values)

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, label='$f(x) = \\frac{e^x}{1 + e^x}$', color='blue')
plt.title('Graph of $f(x) = \\frac{e^x}{1 + e^x}$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()
