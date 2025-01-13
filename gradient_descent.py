import numpy as np

# Sample data (x, y values)
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 1.3, 3.75, 2.25])

# Initialize parameters
theta0 = 1  # Intercept
theta1 = 0.5  # Slope
alpha = 0.1  # Learning rate

# Number of data points
m = len(x)

# Step 1: Compute predictions
y_pred = theta0 + theta1 * x  # Predicted y values

# Step 2: Compute residuals (errors)
residuals = y_pred - y  # Difference between predicted and actual values

# Step 3: Calculate gradients
grad_theta0 = (1 / m) * np.sum(residuals)  # Gradient for theta0
grad_theta1 = (1 / m) * np.sum(residuals * x)  # Gradient for theta1

# Step 4: Update parameters
theta0 -= alpha * grad_theta0  # Update theta0
theta1 -= alpha * grad_theta1  # Update theta1

# Print results
print("After one step of gradient descent:")
print(f"Updated Theta0 (intercept): {theta0}")
print(f"Updated Theta1 (slope): {theta1}")
print(f"Gradient Theta0: {grad_theta0}")
print(f"Gradient Theta1: {grad_theta1}")
