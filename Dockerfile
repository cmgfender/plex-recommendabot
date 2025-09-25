# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script and the .env file into the container
COPY . .

# Set the command to run when the container starts
# Replace 'your_script_name.py' with the actual name of your script
CMD ["python3", "./your_script_name.py"]
