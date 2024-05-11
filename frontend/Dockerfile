FROM python:3.11

# Set working directory to /app
WORKDIR /app

# Copy the frontend directory
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]
