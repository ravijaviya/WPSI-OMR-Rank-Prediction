# 1. Base image
FROM python:3.10-slim

# 2. Install Poppler for PDF processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app's code
COPY . .

# 6. Expose Streamlit's default port
EXPOSE 8501

# 7. Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]