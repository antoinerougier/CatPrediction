from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def download_data_task():
    import sys

    sys.path.insert(0, "/opt/airflow/project")
    from src.data.download_data import main

    main()


def train_model_task(**context):
    import sys

    sys.path.insert(0, "/opt/airflow/project")
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import get_dataloaders
    from src.model import build_model
    from src.train import train_one_epoch, evaluate
    import torch.nn as nn
    import torch.optim as optim
    import mlflow
    import mlflow.pytorch
    import json

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("cat-dog-classifier")

    device = torch.device("cpu")
    num_epochs = 5
    learning_rate = 1e-3
    batch_size = 32

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir="/opt/airflow/project/data", batch_size=batch_size
    )

    model = build_model(num_classes=len(class_names), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model": "resnet18",
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "freeze_backbone": True,
            }
        )

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

        test_loss, test_acc = evaluate(model, test_loader, criterion)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})

        mlflow.pytorch.log_model(
            model, "model", registered_model_name="cat-dog-classifier"
        )

        # Passe le run_id et test_acc à la tâche suivante via XCom
        context["ti"].xcom_push(key="run_id", value=run.info.run_id)
        context["ti"].xcom_push(key="test_acc", value=test_acc)

    # Sauvegarde locale
    import os

    os.makedirs("/opt/airflow/project/models", exist_ok=True)
    torch.save(model.state_dict(), "/opt/airflow/project/models/model.pt")
    with open("/opt/airflow/project/models/class_names.json", "w") as f:
        json.dump({"class_names": class_names}, f)


def check_model_quality(**context):
    """Branch : si test_acc >= seuil → promote, sinon → reject"""
    test_acc = context["ti"].xcom_pull(key="test_acc", task_ids="train_model")
    print(f"Test accuracy : {test_acc:.4f}")
    if test_acc >= 0.90:
        return "promote_model"
    return "reject_model"


def promote_model_task(**context):
    import sys

    sys.path.insert(0, "/opt/airflow/project")
    import mlflow

    mlflow.set_tracking_uri("http://mlflow:5000")
    client = mlflow.MlflowClient()

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
    model_version = client.get_latest_versions("cat-dog-classifier", stages=["None"])[0]

    client.transition_model_version_stage(
        name="cat-dog-classifier",
        version=model_version.version,
        stage="Production",
    )
    print(f"Modèle v{model_version.version} promu en Production ✓")


with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="Pipeline complet : data → train → evaluate → promote",
    schedule_interval=None,  # déclenché manuellement ou par le DAG monitoring
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training"],
) as dag:

    start = EmptyOperator(task_id="start")

    download = PythonOperator(
        task_id="download_data",
        python_callable=download_data_task,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
    )

    check_quality = BranchPythonOperator(
        task_id="check_model_quality",
        python_callable=check_model_quality,
    )

    promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_task,
    )

    reject = EmptyOperator(task_id="reject_model")

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> download >> train >> check_quality >> [promote, reject] >> end
