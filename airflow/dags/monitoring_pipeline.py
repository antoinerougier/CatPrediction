from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def check_drift_task(**context):
    import sys

    sys.path.insert(0, "/opt/airflow/project")
    from datetime import datetime, timedelta
    from sqlalchemy import func
    import mlflow

    mlflow.set_tracking_uri("http://mlflow:5000")

    from monitoring.database import SessionLocal, PredictionLog

    LOOKBACK_DAYS = 30
    MIN_FEEDBACK_COUNT = 20
    ACCURACY_DROP_THRESHOLD = 0.05

    # Baseline depuis MLflow
    client = mlflow.MlflowClient()
    try:
        latest = client.get_latest_versions(
            "cat-dog-classifier", stages=["Production"]
        )[0]
        run = client.get_run(latest.run_id)
        baseline_acc = run.data.metrics.get("test_acc", 0.95)
    except Exception:
        baseline_acc = 0.95  # fallback si pas encore de modèle en prod

    # Live accuracy depuis les feedbacks
    db = SessionLocal()
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    total = (
        db.query(func.count(PredictionLog.id))
        .filter(
            PredictionLog.feedback.isnot(None),
            PredictionLog.timestamp >= cutoff,
        )
        .scalar()
    )
    correct = (
        db.query(func.count(PredictionLog.id))
        .filter(
            PredictionLog.feedback.is_(True),
            PredictionLog.timestamp >= cutoff,
        )
        .scalar()
    )
    db.close()

    print(f"Baseline : {baseline_acc:.4f} | Feedbacks : {correct}/{total}")

    context["ti"].xcom_push(key="baseline_acc", value=baseline_acc)
    context["ti"].xcom_push(key="total_feedback", value=total)

    if total < MIN_FEEDBACK_COUNT:
        print("Pas assez de feedback, pas de calcul de drift.")
        return "no_drift"

    live_acc = correct / total
    context["ti"].xcom_push(key="live_acc", value=live_acc)
    drop = baseline_acc - live_acc

    print(f"Live acc : {live_acc:.4f} | Drop : {drop:.4f}")

    if drop > ACCURACY_DROP_THRESHOLD:
        return "drift_detected"
    return "no_drift"


def send_alert_task(**context):
    baseline_acc = context["ti"].xcom_pull(key="baseline_acc", task_ids="check_drift")
    live_acc = context["ti"].xcom_pull(key="live_acc", task_ids="check_drift")
    total = context["ti"].xcom_pull(key="total_feedback", task_ids="check_drift")

    message = (
        f"⚠️ DRIFT DÉTECTÉ\n"
        f"Baseline accuracy : {baseline_acc:.4f}\n"
        f"Live accuracy : {live_acc:.4f}\n"
        f"Drop : {baseline_acc - live_acc:.4f}\n"
        f"Basé sur {total} feedbacks\n"
        f"→ Ré-entraînement déclenché automatiquement."
    )
    print(message)

    # Log dans un fichier pour l'instant (tu pourras brancher un email/Slack plus tard)
    import os

    os.makedirs("/opt/airflow/project/monitoring/alerts", exist_ok=True)
    alert_path = f"/opt/airflow/project/monitoring/alerts/alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(alert_path, "w") as f:
        f.write(message)
    print(f"Alerte sauvegardée : {alert_path}")


with DAG(
    dag_id="monitoring_pipeline",
    default_args=default_args,
    description="Vérifie le drift quotidiennement et déclenche un ré-entraînement si besoin",
    schedule_interval="0 6 * * *",  # tous les jours à 6h UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "monitoring"],
) as dag:

    start = EmptyOperator(task_id="start")

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift_task,
    )

    drift_detected = EmptyOperator(task_id="drift_detected")
    no_drift = EmptyOperator(task_id="no_drift")

    alert = PythonOperator(
        task_id="send_alert",
        python_callable=send_alert_task,
    )

    # Déclenche automatiquement le pipeline d'entraînement si drift
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> check_drift >> [drift_detected, no_drift]
    drift_detected >> alert >> trigger_retrain >> end
    no_drift >> end
