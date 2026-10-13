from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocess import build_preprocessor, load_and_preprocess_data
from train_utils import train_and_log


def main():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    preprocessor = build_preprocessor()
    train_and_log("Logistic_Regression", clf, X_train, X_test, y_train, y_test, preprocessor)


if __name__ == "__main__":
    main()
