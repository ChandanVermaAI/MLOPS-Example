import mlflow
import psutil
import time
from datetime import datetime

# Set the experiment name
mlflow.set_experiment("Continuous System Monitoring")

# Function to log CPU usage
def log_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    mlflow.log_metric("cpu_usage", cpu_usage)

# Function to log memory usage
def log_memory_usage():
    memory_info = psutil.virtual_memory()
    mlflow.log_metric("total_memory", memory_info.total)
    mlflow.log_metric("available_memory", memory_info.available)
    mlflow.log_metric("used_memory", memory_info.used)
    mlflow.log_metric("memory_usage_percent", memory_info.percent)

# Function to log disk usage
def log_disk_usage():
    disk_usage = psutil.disk_usage('/')
    mlflow.log_metric("total_disk_space", disk_usage.total)
    mlflow.log_metric("used_disk_space", disk_usage.used)
    mlflow.log_metric("free_disk_space", disk_usage.free)
    mlflow.log_metric("disk_usage_percent", disk_usage.percent)

# Function to log network statistics
def log_network_stats():
    network_stats = psutil.net_io_counters()
    mlflow.log_metric("bytes_sent", network_stats.bytes_sent)
    mlflow.log_metric("bytes_received", network_stats.bytes_recv)

# Function to log process details
def log_process_details():
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        with mlflow.start_run(nested=True):
            mlflow.log_param("process_pid", proc.info['pid'])
            mlflow.log_param("process_name", proc.info['name'])
            mlflow.log_param("process_username", proc.info['username'])
            try:
                process = psutil.Process(proc.info['pid'])
                mlflow.log_metric("process_memory_usage", process.memory_info().rss)
                mlflow.log_metric("process_cpu_usage", process.cpu_percent(interval=1))
            except psutil.NoSuchProcess:
                mlflow.log_metric("process_memory_usage", 0)
                mlflow.log_metric("process_cpu_usage", 0)

# Set the logging interval (in seconds)
tracking_interval_seconds = 10

# Start a new MLflow run
with mlflow.start_run():
    mlflow.log_param("tracking_interval_seconds", tracking_interval_seconds)

    while True:  # Infinite loop for continuous monitoring
        # Log timestamp as a parameter
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mlflow.log_param("timestamp", timestamp)
        log_cpu_usage()
        log_memory_usage()
        log_disk_usage()
        log_network_stats()
        log_process_details()
        time.sleep(tracking_interval_seconds)
        print("Run")
