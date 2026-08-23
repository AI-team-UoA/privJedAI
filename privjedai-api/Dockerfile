# Use the official Python slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies first (this layer gets cached, speeding up future builds)
RUN pip install --no-cache-dir uvicorn
RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir pydantic
RUN pip install --no-cache-dir privjedai

# Copy the rest of your application code
COPY . .

# Expose the port Uvicorn will run on
EXPOSE 8000

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]