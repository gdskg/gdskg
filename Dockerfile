# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install git for repository cloning support
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary application code to keep context small and builds fast
# Source code explicitly copied below
COPY main.py .
COPY core/ ./core/
COPY analysis/ ./analysis/
COPY mcp_server/ ./mcp_server/
COPY plugins/ ./plugins/
COPY VERSION .

# Define environment variables
ENV PYTHONUNBUFFERED=1
ENV GDSKG_VECTOR_DB_DIR=/app/vector_db

# Create vector DB directory and define as volume
RUN mkdir -p /app/vector_db
VOLUME ["/app/vector_db"]

ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
ENV GIT_PYTHON_REFRESH=quiet

# Run main.py when the container launches
EXPOSE 8015
ENTRYPOINT ["python", "main.py"]
CMD ["serve", "--transport", "sse", "--port", "8015", "--host", "0.0.0.0"]
