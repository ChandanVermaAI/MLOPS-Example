import mlflow
from mlflow import MlflowClient
# Initialize the MLflow client
client = MlflowClient()
# Create an experiment
experiment_name = "Demo Experiment"
experiment_id = client.create_experiment(experiment_name)
# Start a new run
run = client.create_run(experiment_id)
# Log parameters
client.log_param(run.info.run_id, "param1", 5)
client.log_param(run.info.run_id, "param2", 10)
# Log metrics
client.log_metric(run.info.run_id, "metric1", 0.87)
client.log_metric(run.info.run_id, "metric2", 0.92)
# Log an artifact (e.g., a text file)
with open("output.txt", "w") as f:
    f.write("Hello, world!")
client.log_artifact(run.info.run_id, "output.txt")

# End the run
client.set_terminated(run.info.run_id)


# Fetch the experiment
experiment = client.get_experiment_by_name(experiment_name)
print(f"Experiment: {experiment}")

# Fetch the run
run_id = run.info.run_id
run_info = client.get_run(run_id)
print(f"Run Info: {run_info}")

# Fetch logged parameters and metrics
params = client.get_run(run_id).data.params
metrics = client.get_run(run_id).data.metrics
print(f"Parameters: {params}")
print(f"Metrics: {metrics}")

# List artifacts
artifacts = client.list_artifacts(run_id)
print(f"Artifacts: {artifacts}")

# Download an artifact
client.download_artifacts(run_id, "output.txt", ".")

with open("output.txt", "r") as f:
    content = f.read()
    print(f"Artifact content: {content}")
