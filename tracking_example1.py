from mlflow import log_metric
from random import choice

metric_names=["cpu","ram","disk"]

percentage=[i for i in range(0,600)]

for i in range(40):
    log_metric(choice(metric_names),choice(percentage))