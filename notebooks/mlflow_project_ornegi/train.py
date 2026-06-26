import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    # Veri yükle
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=args.test_size, random_state=42
    )

    with mlflow.start_run():
        # Parametreleri logla
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)

        # Model eğit
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrikleri logla
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Modeli kaydet
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    main()
