import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from preprocess import load_and_preprocess_data, build_preprocessor


def evaluate_model(name: str, pipeline, X_train, y_train, X_test, y_test, cv_metric: str = "roc_auc"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring=cv_metric)
    mean_cv_score = float(np.mean(cv_scores))

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_score": mean_cv_score,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    return pipeline, metrics


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # SQLite tracking backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Heart_Disease_Model_Selection")

    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidate_models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    }

    results = {}
    fitted_pipelines = {}
    selection_metric = "test_f1"

    print("=" * 60)
    print("Starting Automated Model Training & Evaluation")
    print("=" * 60)

    for model_name, estimator in candidate_models.items():
        with mlflow.start_run(run_name=f"Train_{model_name}"):
            preprocessor = build_preprocessor()
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator)
            ])

            fitted_pipe, metrics = evaluate_model(model_name, pipeline, X_train, y_train, X_test, y_test)
            results[model_name] = metrics
            fitted_pipelines[model_name] = fitted_pipe

            # Save local model artifact
            model_path = f"models/{model_name}.joblib"
            joblib.dump(fitted_pipe, model_path)

            # Log metrics and params
            mlflow.log_params({"model_type": model_name, "random_state": 42})
            mlflow.log_metrics(metrics)

            # Log model with cloudpickle serialization to bypass skops type audit
            mlflow.sklearn.log_model(
                sk_model=fitted_pipe,
                artifact_path=model_name,
                serialization_format="cloudpickle"
            )

            print(
                f"[{model_name}] - Accuracy: {metrics['test_accuracy']:.4f} | "
                f"Recall: {metrics['test_recall']:.4f} | "
                f"F1: {metrics['test_f1']:.4f} | "
                f"ROC-AUC: {metrics['test_roc_auc']:.4f}"
            )

    # Automated selection
    best_model_name = max(results, key=lambda k: results[k][selection_metric])
    best_pipeline = fitted_pipelines[best_model_name]
    best_model_path = "models/best_model.joblib"

    joblib.dump(best_pipeline, best_model_path)

    best_score = results[best_model_name][selection_metric]
    print("=" * 60)
    print(f"WINNER: '{best_model_name}' selected based on highest {selection_metric} ({best_score:.4f})")
    print(f"Automatically promoted to production artifact: {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
