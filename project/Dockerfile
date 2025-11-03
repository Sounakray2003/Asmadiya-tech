# ===== Base image =====
FROM python:3.10

# ===== Install system dependencies =====
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# ===== Set working directory =====
WORKDIR /app

# ===== Copy dependency list =====
COPY requirements.txt .

# ===== Install Python dependencies =====
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter

# ===== Copy notebook(s) =====
# Make sure your actual notebook file is in the same directory as this Dockerfile
COPY *.ipynb /app/

# ===== Expose Jupyter port =====
EXPOSE 8888

# ===== Start Jupyter Notebook =====
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
