FROM python:3.12-slim

WORKDIR /app

# Install the package first so dependencies are cached across rebuilds.
# Editable install keeps the source at /app/music, so config.py resolves
# PROJECT_ROOT to /app and finds /app/Assets correctly.
COPY pyproject.toml README.md ./
COPY music/ ./music/
RUN pip install --no-cache-dir -e .

# Bring in the app and raw data.
COPY app/ ./app/
COPY Assets/ ./Assets/

# Build the processed dataset and trained model into the image so the
# dashboard starts instantly (these artifacts are gitignored).
RUN process-data && train-model

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
