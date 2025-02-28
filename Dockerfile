# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code from your working directory into the container at /app
COPY app.py /app
COPY train.py /app
COPY datos.txt /app
COPY test_app.py /app

# Train the model
RUN python train.py

# Define environment variable
ENV PYTHONPATH=/app

# Run the tests
CMD gunicorn app:app --bind 0.0.0.0:5000 & \
    python -m pytest -v test_app.py

# Expose port 5000 to the outside world
EXPOSE 5000
