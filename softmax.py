import numpy as np

def softmax(logits):
    # Subtracting the max value from logits for numerical stability
    # This ensures that the largest logit value is 0, preventing large exponentials
    logits = np.array(logits)  # Convert to numpy array if not already
    exp_logits = np.exp(logits - np.max(logits))  # Exponentiate logits
    return exp_logits / np.sum(exp_logits)  # Normalize to get probabilities

# Example Usage:
logits = [2.0, 1.0, 0.1]  # Example raw predictions (logits)
probabilities = softmax(logits)
print("Softmax Probabilities:", probabilities)