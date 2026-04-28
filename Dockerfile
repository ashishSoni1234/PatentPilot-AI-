FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for compiling some python packages and FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Expose ports (Hugging Face routes traffic to 7860)
EXPOSE 7860
EXPOSE 8000

# Run the start script
CMD ["./start.sh"]
