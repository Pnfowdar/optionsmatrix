# Dockerfile

FROM python:3.11-slim

# Set the environment variable early
ENV PYTHONDONTWRITEBYTECODE 1
# Set Python output to be unbuffered (good for logs)
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# (Now respects .dockerignore, so __pycache__ etc. are excluded)
COPY . .

EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]