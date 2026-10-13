import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
EXPERIMENT_NAME = "Heart_Disease_Prediction_CI_CD"


def train_and_log(
    model_name: str, classifier, X_train, X_test, y_train, y_test, preprocessor, artifacts_dir: str = "artifacts"
):
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    with mlflow.start_run(run_name=model_name):
        print(f"\n==================== Training {model_name} ====================")

        # 5-Fold Stratified Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=skf, scoring="roc_auc")
        cv_mean_roc_auc = float(np.mean(cv_scores))
        cv_std_roc_auc = float(np.std(cv_scores))

        # Fit on full training data
        full_pipeline.fit(X_train, y_train)

        # Predict on holdout test set
        y_pred = full_pipeline.predict(X_test)
        y_proba = full_pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        roc_auc = float(roc_auc_score(y_test, y_proba))

        print(f"Results for {model_name}:")
        print(f"  CV ROC-AUC (mean +/- std): {cv_mean_roc_auc:.4f} (+/- {cv_std_roc_auc:.4f})")
        print(f"  Test Accuracy           : {acc:.4f}")
        print(f"  Test Precision          : {prec:.4f}")
        print(f"  Test Recall             : {rec:.4f}")
        print(f"  Test F1-Score           : {f1:.4f}")
        print(f"  Test ROC-AUC            : {roc_auc:.4f}")

        # MLflow Parameters
        mlflow.log_param("model_name", model_name)
        for param, val in classifier.get_params().items():
            mlflow.log_param(f"model__{param}", str(val))

        # MLflow Metrics
        mlflow.log_metric("cv_mean_roc_auc", cv_mean_roc_auc)
        mlflow.log_metric("cv_std_roc_auc", cv_std_roc_auc)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        # Artifact 1: Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Heart Disease"]).plot(ax=ax_cm, cmap="Blues")
        ax_cm.set_title(f"Confusion Matrix - {model_name}")
        cm_path = os.path.join(artifacts_dir, f"{model_name}_cm.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path)

        # Artifact 2: ROC Curve
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_estimator(full_pipeline, X_test, y_test, ax=ax_roc)
        ax_roc.set_title(f"ROC Curve - {model_name}")
        roc_path = os.path.join(artifacts_dir, f"{model_name}_roc.png")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close(fig_roc)
        mlflow.log_artifact(roc_path)

        # MLflow Model Serialization
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            name="model",
            serialization_format="cloudpickle"
        )

        # Standalone local joblib model
        model_save_path = os.path.join("models", f"{model_name}.joblib")
        joblib.dump(full_pipeline, model_save_path)
        print(f"  Model artifact saved to: {model_save_path}")

    return full_pipeline, roc_auc
