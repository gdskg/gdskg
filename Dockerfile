# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make port 80 available to the world outside this container
# (Though MCP usually runs over stdio or SSE/HTTP, depending on config)
# exposing port isn't strictly necessary for stdio but good practice if we add HTTP later.
# EXPOSE 8000 

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run main.py when the container launches
# Default to running the SSE server on port 8015
EXPOSE 8015
ENTRYPOINT ["python", "main.py"]
CMD ["serve", "--transport", "sse", "--port", "8015", "--host", "0.0.0.0"]
