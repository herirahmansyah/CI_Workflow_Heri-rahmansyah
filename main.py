import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

def load_preprocessed_data():
    """Memuat data yang telah dipreproses."""
    # Menyesuaikan path untuk dijalankan di dalam container docker
    # Data diasumsikan diletakkan di dalam folder data saat di-copy ke container
    train_path = 'data/train_df.csv'
    test_path = 'data/test_df.csv'

    if not os.path.exists(train_path) or not os.path.exists(test_path):
         print(f"Error: Data tidak ditemukan. Pastikan file train_df.csv dan test_df.csv ada di folder 'data/'")
         sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Attrition'])
    y_train = train_df['Attrition']
    X_test = test_df.drop(columns=['Attrition'])
    y_test = test_df['Attrition']

    return X_train, y_train, X_test, y_test

def create_confusion_matrix_plot(y_test, y_pred, save_path="confusion_matrix.png"):
    """Membuat dan menyimpan plot Confusion Matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    plt.close()
    return save_path

def create_roc_curve_plot(y_test, y_probs, save_path="roc_curve.png"):
    """Membuat dan menyimpan plot ROC Curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    return save_path

def main():
    print("Memulai proses pemodelan dari Docker Container...")

    # 1. Load Data
    X_train, y_train, X_test, y_test = load_preprocessed_data()

    # Inisialisasi DagsHub untuk Remote Tracking
    # Pastikan variable DAGSHUB_USER_TOKEN sudah di set di environment
    try:
        dagshub.init(repo_owner='herirahmansyah', repo_name='Eksperimen_SML_Heri-rahmansyah', mlflow=True)
    except Exception as e:
        print(f"Warning: Gagal menginisialisasi DagsHub ({e}). Proses akan dilanjutkan dengan MLflow lokal (jika tracking URI tidak di-set via ENV).")

    # Set MLflow Experiment (setelah inisialisasi remote server)
    mlflow.set_experiment("HR_Attrition_Prediction_RF_Docker")

    # Mulai run MLflow
    with mlflow.start_run(run_name="RandomForest_DockerRun"):
        print("MLflow run dimulai...")

        # 2. Definisikan Model dan Hyperparameter Grid
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

        # 3. Tuning Hyperparameter menggunakan GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        print("Melakukan GridSearchCV...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # 4. Prediksi dan Evaluasi
        y_pred = best_model.predict(X_test)
        y_probs = best_model.predict_proba(X_test)[:, 1] # Probabilitas kelas positif

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        print("Parameter terbaik:", best_params)
        print("Metrik Evaluasi:", metrics)

        # 5. Logging ke MLflow (Manual Logging)
        # Log parameter terbaik
        mlflow.log_params(best_params)

        # Log metrik
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(best_model, "random_forest_model")

        # 6. Pembuatan dan Logging Artefak Gambar
        print("Membuat plot artefak...")
        os.makedirs("artifacts", exist_ok=True)
        cm_path = create_confusion_matrix_plot(y_test, y_pred, save_path="artifacts/confusion_matrix.png")
        roc_path = create_roc_curve_plot(y_test, y_probs, save_path="artifacts/roc_curve.png")

        # Log artefak ke MLflow
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        print(f"Run MLflow selesai. Model, parameter, metrik, dan artefak telah dicatat.")

if __name__ == "__main__":
    main()
