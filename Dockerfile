FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt update -y && apt install -y \
    libpoppler-cpp-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]

