# Use the official Python 3.13 image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy the contents of the functions folder into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]