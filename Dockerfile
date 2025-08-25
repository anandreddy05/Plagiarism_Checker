# Base image
FROM python:3.12-slim

# Working dir
WORKDIR /app

# Copy everything
COPY . /app

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 5000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
