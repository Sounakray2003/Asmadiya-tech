# # Dockerfile (builds a Triton server image with the Python backend and TTS deps)
# # NOTE: You must have Docker and (for GPU) NVIDIA Container Toolkit installed.

# ARG TRITON_IMAGE=nvcr.io/nvidia/tritonserver:23.09-py3
# FROM ${TRITON_IMAGE}

# # Install system packages
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         ffmpeg \
#         libsndfile1 \
#         git \
#         ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# # Copy model repository into container
# COPY model_repository /models

# # Upgrade pip then install python deps
# RUN python3 -m pip install --upgrade pip setuptools wheel

# # Important: pick correct torch wheel for your CUDA. For CPU-only usage default pip 'torch' will install CPU build.
# # If you HAVE NVIDIA GPU and want CUDA acceleration, replace the following pip install with an explicit wheel:
# # e.g. for CUDA 12.1 (example) you might run:
# # RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.0+cu121
# #
# # Here we install general packages; if you need a specific torch+cuda, update below before building.
# COPY requirements.txt .
# RUN pip install -r requirements.txt 

# # Expose Triton ports
# EXPOSE 8000 8001 8002

# # Start Triton server with the model repository mounted at /models
# ENTRYPOINT ["/opt/tritonserver/bin/tritonserver", "--model-repository=/models"]







# FROM nvcr.io/nvidia/tritonserver:23.10-py3
 
# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libsndfile1 \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*
 
# # Set working directory
# WORKDIR /workspace
# COPY requirements.txt .
 
# RUN pip install --no-cache-dir -r requirements.txt
# # Copy model repository
# COPY model_repository /models
 
# # Install Python dependencies
# # RUN pip install --no-cache-dir \
# #     transformers>=4.35.0 \+
 
# #     soundfile \
# #     torch \
# #     huggingface_hub
 
# # Create a startup script
# RUN echo '#!/bin/bash\n\
# \n\
# # Login to Hugging Face (optional - you can set HF_TOKEN as environment variable)\n\
# if [ ! -z "$HF_TOKEN" ]; then\n\
#     python -c "from huggingface_hub import login; login(token=\"$HF_TOKEN\")"\n\
# fi\n\
# \n\
# # Start Triton server\n\
# exec tritonserver --model-repository=/models --log-verbose=1 --strict-model-config=false' > /usr/local/bin/start.sh
 
# RUN chmod +x /usr/local/bin/start.sh
 
# # Expose Triton ports
# EXPOSE 8000 8001 8002
 
# CMD ["/usr/local/bin/start.sh"]
 
# ==============================================================
# Stage 1: Use NVIDIA Triton base image with Python backend
# ==============================================================
 
FROM nvcr.io/nvidia/tritonserver:24.09-py3
 
# Set working directory inside container
WORKDIR /workspace
 
# --------------------------------------------------------------
# Install Python dependencies
# --------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*
 
# Install required Python packages
RUN pip install --no-cache-dir --ignore-installed \
    torch \
    transformers \
    soundfile \
    numpy \
    huggingface-hub \
    TTS \
    sentencepiece \
    accelerate

# --------------------------------------------------------------
# Copy the Triton model repository
# --------------------------------------------------------------
COPY model_repository /models
 
# Expose Triton HTTP and gRPC ports
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002
 
# --------------------------------------------------------------
# Run Triton Inference Server
# --------------------------------------------------------------
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
 
# docker build -t <image_name> .
# docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 -e HF_TOKEN=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 indic_tts
