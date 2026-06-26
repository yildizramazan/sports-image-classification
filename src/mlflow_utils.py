import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def setup_mlflow(config):
    """Sets up MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment = mlflow.set_experiment(config["mlflow"]["experiment_name"])
    return experiment

def log_params_from_config(config):
    """Flattens config and logs parameters to MLflow."""
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_config = flatten_dict(config)
    for k, v in flat_config.items():
        mlflow.log_param(k, v)

def log_training_metrics(epoch, train_loss, train_acc, val_loss, val_acc):
    """Logs training metrics per epoch."""
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
    mlflow.log_metric("val_accuracy", val_acc, step=epoch)

def log_model_artifact(model, model_name, input_example=None):
    """Logs PyTorch model as MLflow artifact."""
    mlflow.pytorch.log_model(model, model_name, input_example=input_example)

def log_plot_artifact(figure, filename):
    """Saves figure to a file and logs as artifact."""
    filepath = filename
    figure.savefig(filepath)
    mlflow.log_artifact(filepath)

def log_confusion_matrix(y_true, y_pred, class_names):
    """Creates and logs a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

def register_best_model(run_id, model_name, model_uri):
    """Registers a model to the Model Registry."""
    mlflow.register_model(model_uri, model_name)

def get_best_run(experiment_name, metric="metrics.val_accuracy"):
    """Finds the best run in an experiment based on a given metric."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=[f"{metric} DESC"])
    if runs.empty:
        return None
    best_run = runs.iloc[0]
    return best_run
