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

best_accuracy = 0
best_model = None
best_run_id = None
mlflow.sklearn.autolog()
# Run experiments with different k values
for k in k_values:
    custom_run_name = "knn_" + str(k)
    with mlflow.start_run(run_name=custom_run_name):
        # Initialize the KNeighborsClassifier with the current k value
        model = KNeighborsClassifier(n_neighbors=k)
        
        # Log the hyperparameter k
        #mlflow.log_param("n_neighbors", k)
        
        # Train the model
        model.fit(X_train, Y_train)
        
        # Make predictions and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions)
        
        # Log the accuracy metric
        #mlflow.log_metric("accuracy", accuracy)
        
        # Check if this model is the best one
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_run_id = mlflow.active_run().info.run_id
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Logged run with n_neighbors={k} and accuracy={accuracy}")

# Register the best model in the model registry
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, "BMI_KNN_Model")
