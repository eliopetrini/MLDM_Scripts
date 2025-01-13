import numpy as np

# Example multi-class confusion matrix
confusion_matrix = np.array([
    [12, 2, 17],  # Class 0: True Positives and False Negatives/Positives
    [3, 8, 20],  # Class 1
    [4, 1, 143]   # Class 2
])
# Actual values -> vertical (12, 3, 4)
# Predicted values -> horizontal


# Total number of classes
num_classes = confusion_matrix.shape[0]

# Initialize variables to store results
accuracy = 0
error = 0
precision_per_class = []
recall_per_class = []
f1_score_per_class = []

# Calculate overall accuracy and error
total_correct = np.trace(confusion_matrix)  # Sum of diagonal elements (True Positives for all classes)
total_samples = np.sum(confusion_matrix)    # Total samples
accuracy = total_correct / total_samples
error = 1 - accuracy

# Calculate precision, recall, and F1-score for each class
for i in range(num_classes):
    true_positive = confusion_matrix[i, i]
    false_positive = np.sum(confusion_matrix[:, i]) - true_positive
    false_negative = np.sum(confusion_matrix[i, :]) - true_positive

    # Precision: TP / (TP + FP)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    precision_per_class.append(precision)

    # Recall: TP / (TP + FN)
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    recall_per_class.append(recall)

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_score_per_class.append(f1_score)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Error: {error:.2f}")
print("Class-wise Metrics:")
for i in range(num_classes):
    print(f"Class {i}:")
    print(f"  Precision: {precision_per_class[i]:.2f}")
    print(f"  Recall: {recall_per_class[i]:.2f}")
    print(f"  F1-Score: {f1_score_per_class[i]:.2f}")
