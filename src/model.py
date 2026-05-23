import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def summarize_dataset(df: pd.DataFrame) -> dict:
    summary = {
        "jumlah_data": df.shape[0],
        "jumlah_fitur": df.shape[1] - 1,
        "fitur": df.columns[:-1].tolist(),
        "target": df.columns[-1],
        "distribusi_kelas": df[df.columns[-1]].value_counts().to_dict(),
    }
    return summary


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    columns_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_clean[columns_with_zero] = df_clean[columns_with_zero].replace(0, np.nan)
    df_clean[columns_with_zero] = df_clean[columns_with_zero].fillna(df_clean[columns_with_zero].median())
    return df_clean


def preprocess_dataset(df: pd.DataFrame):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"].astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, scaler


def split_dataset(X: np.ndarray, y: np.ndarray, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_ann_model(input_dim: int, learning_rate=0.01) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=(16, 12, 8),
        activation="relu",
        solver="sgd",
        learning_rate_init=learning_rate,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20,
        verbose=False,
    )
    return model


def train_model(model: MLPClassifier, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: MLPClassifier, X_test, y_test) -> dict:
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def save_artifacts(model: MLPClassifier, scaler: StandardScaler, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "diabetes_ann.pkl"
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


def plot_training_history(model: MLPClassifier, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model.loss_curve_, label="Training Loss")
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    path = output_dir / "training_history.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    path = output_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_distribution(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_columns = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
    fig, axes = plt.subplots(len(selected_columns), 1, figsize=(8, 20))
    for idx, column in enumerate(selected_columns):
        sns.histplot(data=df, x=column, hue="Outcome", kde=True, ax=axes[idx], palette="Set2", alpha=0.6)
        axes[idx].set_title(f"Distribusi {column}")
    path = output_dir / "feature_distribution.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_evaluation_report(metrics: dict, summary: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = (
        f"Jumlah data: {summary['jumlah_data']}\n"
        f"Jumlah fitur: {summary['jumlah_fitur']}\n"
        f"Distribusi kelas: {summary['distribusi_kelas']}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1-score: {metrics['f1_score']:.4f}\n"
    )
    path = output_dir / "evaluation_report.txt"
    with open(path, "w", encoding="utf-8") as file:
        file.write(report)
    return path
