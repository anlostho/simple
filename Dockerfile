# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code from your working directory into the container at /app
COPY generado.py /app
COPY datos.txt /app

# Define environment variable
ENV PYTHONPATH=/app

# Run generado.py when the container launches
CMD ["python", "generado.py"]
