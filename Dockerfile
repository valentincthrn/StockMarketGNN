FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch Geometric and other Python dependencies
RUN pip install torch-geometric numpy pandas scikit-learn jupyter click mlflow yfinance tqdm DadosAbertosBrasil

COPY . /workspace

WORKDIR /workspace