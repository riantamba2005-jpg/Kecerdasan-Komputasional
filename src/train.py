import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from src.model import (
    build_ann_model,
    clean_dataset,
    evaluate_model,
    load_dataset,
    plot_confusion_matrix,
    plot_feature_distribution,
    plot_training_history,
    save_artifacts,
    save_evaluation_report,
    split_dataset,
    summarize_dataset,
    train_model,
    preprocess_dataset,
)


def main():
    root = Path(__file__).resolve().parent.parent
    dataset_path_candidates = [root / "data" / "pima_diabetes.csv", root / "data" / "diabetes.csv"]
    dataset_path = next((path for path in dataset_path_candidates if path.exists()), None)

    if dataset_path is None:
        print("Dataset tidak ditemukan.")
        print("Silakan letakkan file Kaggle Pima Indians Diabetes Dataset sebagai data/pima_diabetes.csv atau data/diabetes.csv")
        return

    df = load_dataset(dataset_path)
    summary = summarize_dataset(df)
    cleaned = clean_dataset(df)
    X, y, scaler = preprocess_dataset(cleaned)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = build_ann_model(input_dim=X_train.shape[1], learning_rate=0.01)
    model = train_model(model, X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    model_dir = root / "models"
    report_dir = root / "reports"
    save_artifacts(model, scaler, model_dir)
    plot_training_history(model, report_dir)
    plot_confusion_matrix(metrics["confusion_matrix"], report_dir)
    plot_feature_distribution(cleaned, report_dir)
    save_evaluation_report(metrics, summary, report_dir)

    print("Training selesai.")
    print(f"Model tersimpan di: {model_dir}")
    print(f"Laporan evaluasi tersimpan di: {report_dir}")
    print("Metrik evaluasi:")
    print(f"- Accuracy : {metrics['accuracy']:.4f}")
    print(f"- Precision: {metrics['precision']:.4f}")
    print(f"- Recall   : {metrics['recall']:.4f}")
    print(f"- F1-score : {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
