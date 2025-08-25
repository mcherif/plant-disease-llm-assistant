FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/tmp
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install pipdeptree for debugging dependencies
RUN pip install pipdeptree

# Debugging: Save dependency tree and installed packages
RUN pipdeptree > /app/dependency-tree.txt
RUN pip list > /app/installed-packages.txt

# Debug: Output Python version and environment variables
RUN python --version > /app/python-version.txt
RUN env > /app/env.txt

# Ensure multipart is removed (if needed)
RUN pip uninstall -y multipart || true

# Copy model and app code
COPY models/vit-finetuned ./models/vit-finetuned

# Copy assets
COPY ./images/plant-disease-logo.png ./images/plant-disease-logo.png
COPY ./README.md ./README.md

# Copy other source code
COPY ./src ./src

# Expose ports for Streamlit (7860) and FastAPI (8000)
EXPOSE 7860
#EXPOSE 8000

# Default to Streamlit; override in docker run if you want FastAPI
# to run FastAPI, we need to change the Dockerfile CMD and ensure the FastAPI app is set up to serve index.html and /predict/.
# Debug: On container start, print environment info before running the app
#CMD python -c "import sys, os; print('Python:', sys.version); print('Env:', dict(os.environ));" && streamlit run app.py --server.port=7860 --server.address=0.0.0.0

# Default to Streamlit; override CMD for FastAPI if needed
#CMD python -c "import sys, os; print('Python:', sys.version); print('Env:', dict(os.environ)); os.system('ls -lR /app');" && streamlit run src/streamlit_app.py --server.port=7860 --server.address=0.0.0.0

#CMD streamlit run src/streamlit_app.py --server.port=7860 --server.address=0.0.0.0
# To run FastAPI, comment out the above CMD and uncomment below:
#CMD python -c "import sys, os; print('Python:', sys.version); print('Env:', dict(os.environ)); os.system('ls -lR /app');" && uvicorn src.inference.app_fastapi:app --host 0.0.0.0 --port 7860
CMD python src/app_gradio.py