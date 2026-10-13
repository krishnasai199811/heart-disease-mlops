import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def load_and_preprocess_data(data_path: str = "data/raw/heart_disease.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data_loader.py first.")

    df = pd.read_csv(data_path)

    # UCI raw dataset uses 'num', processed versions often use 'target'
    target_col = "target" if "target" in df.columns else "num"

    # Binarize target: 0 = healthy, >0 = heart disease
    df["target"] = (df[target_col] > 0).astype(int)

    # Drop raw target columns if differing
    cols_to_drop = [c for c in ["target", "num"] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df["target"]
    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print(f"Data successfully loaded. Features shape: {X.shape}, Target shape: {y.shape}")
