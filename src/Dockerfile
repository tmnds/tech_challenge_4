# Use an official Python runtime as a parent image
FROM python:3.12.2-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add current directory code to /app in container
ADD src/ /app

# Install pipenv
RUN pip install --no-cache-dir -r ./requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8080

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080","--reload"]