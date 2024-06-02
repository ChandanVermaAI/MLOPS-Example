import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import mlflow.models.evaluation
import shap
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
# Load the dataset
bmi_data = pd.read_csv(r"Bmi_male_female.csv")
X_feature = bmi_data.iloc[:, 0:3]
Y_target = bmi_data.iloc[:, 3]
# Enable autologging for scikit-learn
mlflow.sklearn.autolog()
# Map Gender to numerical values
X_feature["Gender"] = X_feature["Gender"].map({"Male": 0, "Female": 1})

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_feature, Y_target, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,Y_train)
predictions = model.predict(X_train)
signature=infer_signature(X_train,predictions)
eval_data=X_test
eval_data["target"]=Y_test
with mlflow.start_run(run_name="knn_eval_exampe") as run:
    model_info = mlflow.sklearn.log_model(model, "model", signature=signature)
    result = mlflow.evaluate(
        model_info.model_uri,
        eval_data,
        targets="target",
        model_type="classifier",
        evaluators=["default"],
    )
