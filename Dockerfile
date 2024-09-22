FROM python:3.9-slim


ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    libpoppler-dev \
    gcc \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir streamlit \
    langchain-community \
    langchain-chroma \
    langchain_ollama \
    ollama \
    pypdf

# Copy the current directory contents into the container
COPY . .

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
