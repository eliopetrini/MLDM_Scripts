def calculate_parameters(k, p):
    # Number of weights (for each layer pair)
    weights = k * p * p + p * p  # k hidden layers + 1 output layer
    # Number of biases (for all layers except input)
    biases = (k + 1) * p
    # Total number of parameters
    total_parameters = weights + biases
    return total_parameters

# Example for k=3 hidden layers and p=4 neurons in each layer
k = 3
p = 4

total_params = calculate_parameters(k, p)
print(f"Total number of parameters: {total_params}")



# Suppose the network has 784
#  inputs, 16
#  nodes in 2
#  hidden layers and 10
#  nodes in the output layer.
#
# The amount of parameters (meaning weights and bias that make up the cost function) is then:
#
# For the weights:
# 784×16+16×16+16×10=12960
#
# For the bias components:
#
# We have 32
#  neurons in the hidden layers and 10
#  in the output, so we have
# 32+10=42
# biases.

