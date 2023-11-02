# Use a base image with Python and development tools
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the Flask application files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 8501

# Run the streamlit application
CMD ["streamlit", "run", "annual_report_analyzer.py"]