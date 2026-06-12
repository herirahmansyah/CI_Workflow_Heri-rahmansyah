FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Salin requirements.txt terlebih dahulu agar Docker bisa melakukan cache pada layer instalasi
COPY requirements.txt .

# Instal dependensi
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh file proyek dan data ke dalam container
COPY . /app

# (Opsional) Tetapkan environment variables MLflow agar DagsHub bisa melacak hasil tracking
# ENV MLFLOW_TRACKING_URI="https://dagshub.com/herirahmansyah/Eksperimen_SML_Heri-rahmansyah.mlflow"
# ENV MLFLOW_TRACKING_USERNAME="herirahmansyah"
# ENV MLFLOW_TRACKING_PASSWORD="YOUR_DAGSHUB_TOKEN_HERE"

# Command default jika tidak di-override oleh MLflow Project
CMD ["python", "main.py"]