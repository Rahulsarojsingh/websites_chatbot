# Use an official Python base image (lightweight version)
FROM python:3.10.2-slim

# Install dependencies for building and sqlite3
RUN apt-get update && \
    apt-get install -y \
    wget \
    build-essential \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Manually install the latest version of SQLite if necessary
RUN wget https://sqlite.org/2023/sqlite-autoconf-3400000.tar.gz && \
    tar -xvzf sqlite-autoconf-3400000.tar.gz && \
    cd sqlite-autoconf-3400000 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3400000.tar.gz sqlite-autoconf-3400000

# Set working directory
WORKDIR /app

COPY . /app

# Copy the requirements.txt (or environment setup) into the container

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
