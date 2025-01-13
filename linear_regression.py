import numpy as np
import matplotlib.pyplot as plt

# the higher r2, the better (points closer to line)

# Sample data stored in a 2D array (columns: x and y)
data = np.array([
    [1, 1],
    [2, 2],
    [3, 1.3],
    [4, 3.75],
    [5, 2.25]
])
# Separate data into x and y
x = data[:, 0]
y = data[:, 1]

# Perform linear regression calculations
n = len(x)

# Calculate means of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate theta1 (slope) and theta2 (intercept)
# theta1 = covariance(x, y) / variance(x)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)
theta1 = numerator / denominator
theta0 = y_mean - theta1 * x_mean

# Print the parameters
print(f"Theta0 (intercept): {theta0}")
print(f"Theta1 (slope): {theta1}")


# Predicted y values based on the regression line
y_pred = theta1 * x + theta0

# Calculate residuals
residuals = y - y_pred

# Calculate R-squared
ss_res = np.sum(residuals**2)  # Sum of squares of residuals
ss_tot = np.sum((y - y_mean)**2)  # Total sum of squares
r2 = 1 - (ss_res / ss_tot)

# Print R-squared, SSres, and SStot
print(f"SSres (Sum of squares of residuals): {ss_res}")
print(f"SStot (Total sum of squares): {ss_tot}")
print(f"R-squared: {r2}")

# Print sum of residuals for every point
for i, res in enumerate(residuals):
    print(f"Residual for point {i + 1} (x={x[i]}, y={y[i]}): {res}")

# Plot the data points and the regression line
plt.figure(figsize=(10, 5))

# Subplot 1: Regression line
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Regression line')

# Add labels, legend, and title
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Linear Regression')
plt.legend()

# Subplot 2: Residual plot
plt.subplot(1, 2, 2)
plt.scatter(x, residuals, color='purple', label='Residuals')
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# Add labels, legend, and title
plt.xlabel('X values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()