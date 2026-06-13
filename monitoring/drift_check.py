import sys
from datetime import datetime, timedelta

import mlflow
from sqlalchemy import func

from monitoring.database import SessionLocal, PredictionLog

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "cat-dog-classifier"
ACCURACY_DROP_THRESHOLD = 0.05  # alerte si baisse de plus de 5 points
MIN_FEEDBACK_COUNT = (
    20  # nombre minimum de feedbacks pour que le calcul soit significatif
)
LOOKBACK_DAYS = 30


def get_baseline_accuracy():
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(
        MODEL_NAME, stages=["Production", "None"]
    )[0]
    run = client.get_run(latest_version.run_id)
    return run.data.metrics.get("test_acc")


def get_live_accuracy():
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
    return correct, total


def main():
    baseline_acc = get_baseline_accuracy()
    correct, total = get_live_accuracy()

    print(f"Baseline accuracy (entraînement) : {baseline_acc:.4f}")
    print(
        f"Feedback disponible : {correct}/{total} sur les {LOOKBACK_DAYS} derniers jours"
    )

    if total < MIN_FEEDBACK_COUNT:
        print(
            f"Pas assez de feedback ({total} < {MIN_FEEDBACK_COUNT}), pas de calcul de drift pour l'instant."
        )
        sys.exit(0)

    live_acc = correct / total
    print(f"Live accuracy : {live_acc:.4f}")

    drop = baseline_acc - live_acc

    if drop > ACCURACY_DROP_THRESHOLD:
        print(
            f"⚠️ ALERTE : la performance a chuté de {drop:.4f} (seuil: {ACCURACY_DROP_THRESHOLD})"
        )
        print("Un ré-entraînement est recommandé.")
        sys.exit(1)  # code de sortie non-nul -> utile pour CI/CD
    else:
        print("Pas de drift détecté, le modèle est toujours performant.")
        sys.exit(0)


if __name__ == "__main__":
    main()
