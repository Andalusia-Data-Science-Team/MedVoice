FROM ghcr.io/astral-sh/uv:0.9.4-python3.11-bookworm

# Set working directory inside container
WORKDIR /app

# Install system dependencies, including ffmpeg and ffprobe
RUN apt-get update && apt-get install -y \
    build-essential \
    libavif-dev pkg-config \
    libjpeg-dev \
    gcc unzip zip \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from src
COPY ./src/requirements.txt .

# Install Python dependencies
RUN uv pip install -r requirements.txt --system

# Copy the entire src folder contents into /app
COPY ./src /app

# Command to run FastAPI app
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]