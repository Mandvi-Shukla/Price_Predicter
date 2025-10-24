import numpy as np
from joblib import load
import os
os.chdir(r"c:\mandvi\machine learning\final project")

# Load the trained model
model = load('best_model.pkl')

# Input features as a 2D array (1 sample, 14_features)
input_features = np.array([[-0.40145693, 1.09019387, -1.45764994, -0.26761547, -0.9960897,
                            1.36865068, -0.75834524, 1.57302338, -1.00376792, -0.75924327,
                            -1.40166207, 0.4320689, -1.04133412, 1.0901]])

# Make prediction
predicted_feature = model.predict(input_features)

# Print result
print(predicted_feature)
