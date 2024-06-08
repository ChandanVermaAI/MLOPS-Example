import mlflow
from mlflow import MlflowClient
# Initialize the MLflow client
client = MlflowClient()

# Create an experiment
experiment_name = "Test Experiment 1"
experiment_id = client.create_experiment(experiment_name)

# Start a new run
run = client.create_run(experiment_id)

# Log parameters
client.log_param(run.info.run_id, "param1", 5)
client.log_param(run.info.run_id, "param2", 10)
client.log_param(run.info.run_id, "param3", 15)
client.log_param(run.info.run_id, "param4", 11)
# Log metrics
client.log_metric(run.info.run_id, "metric1", 0.87)
client.log_metric(run.info.run_id, "metric2", 0.92)

# Log an artifact (e.g., a text file)
with open("output.txt", "w") as f:
    f.write("Hello, world!")

client.log_artifact(run.info.run_id, "output.txt")

# End the run
client.set_terminated(run.info.run_id)
