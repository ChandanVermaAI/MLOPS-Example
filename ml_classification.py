import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Load the dataset
bmi_data = pd.read_csv(r"Bmi_male_female.csv")
X_feature = bmi_data.iloc[:, 0:3]
Y_target = bmi_data.iloc[:, 3]

# Map Gender to numerical values
X_feature["Gender"] = X_feature["Gender"].map({"Male": 0, "Female": 1})

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_feature, Y_target, test_size=0.2, random_state=42)

# Define different values for the hyperparameter k (number of neighbors)
k_values = [3, 5, 7, 9, 11]
mlflow.set_experiment("BMI Classification")
# Run experiments with different k values
for k in k_values:
    with mlflow.start_run():
        # Initialize the KNeighborsClassifier with the current k value
        model = KNeighborsClassifier(n_neighbors=k)
        
        # Log the hyperparameter k
        mlflow.log_param("n_neighbors", k)
        
        # Train the model
        model.fit(X_train, Y_train)
        
        # Make predictions and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions)
        
        # Log the accuracy metric
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Logged run with n_neighbors={k} and accuracy={accuracy}")

# Print details of all runs
print("Run details:")
for run in mlflow.search_runs():
    print(run)
