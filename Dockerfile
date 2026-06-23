# Base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# System deps (if you hit issues with numpy/pandas/etc you can add build-essential here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command: run the API
# Adjust workers if you want, but 1 is fine for the capstone
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
