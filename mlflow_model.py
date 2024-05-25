import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
mlflow.set_experiment("Logistic Regression")

# Load the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Define different sets of parameters for the experiments
param_sets = [
    {"max_iter": 100, "solver": "lbfgs"},
    {"max_iter": 200, "solver": "liblinear"},
    {"max_iter": 300, "solver": "saga"}
]

# Run experiments with different parameters
for params in param_sets:
    with mlflow.start_run():
        # Initialize the model with the current parameters
        model = LogisticRegression(max_iter=params["max_iter"], solver=params["solver"])
        
        # Log parameters
        mlflow.log_param("max_iter", params["max_iter"])
        mlflow.log_param("solver", params["solver"])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Logged run with params {params} and accuracy {accuracy}")

# Print details of all runs
print("Run details:")
for run in mlflow.search_runs():
    print(run)
