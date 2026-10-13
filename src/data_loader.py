import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_uci_heart_disease(output_dir: str = "data/raw") -> str:
    """Fetches the Heart Disease dataset from UCI ML Repository and saves it as a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "heart_disease.csv")

    print("Fetching Heart Disease dataset from UCI repository (ID: 45)...")
    try:
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets

        # Combine features and target into a single DataFrame
        df = pd.concat([X, y], axis=1)

        df.to_csv(csv_path, index=False)
        print(f"Dataset successfully saved to: {csv_path}")
        print(f"Dataset shape: {df.shape}")
        return csv_path
    except Exception as e:
        print(f"ucimlrepo fetch failed: {e}. Attempting direct fallback download...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope",
            "ca", "thal", "num"
        ]
        df = pd.read_csv(url, names=columns, na_values="?")
        df.to_csv(csv_path, index=False)
        print(f"Fallback download successful. Saved to: {csv_path}")
        print(f"Dataset shape: {df.shape}")
        return csv_path


if __name__ == "__main__":
    download_uci_heart_disease()
