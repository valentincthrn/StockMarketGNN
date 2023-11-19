FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch Geometric and other Python dependencies
RUN pip install torch-geometric numpy pandas scikit-learn jupyter click mlflow yfinance tqdm DadosAbertosBrasil black streamlit

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY . /workspace

WORKDIR /workspace

ENTRYPOINT ["streamlit", "run", "st_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]

