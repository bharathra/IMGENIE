FROM rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0

RUN apt-get update && apt-get install -y dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers.git \
    transformers>=4.38.0 \
    pillow>=9.0.0 \
    accelerate>=0.20.0 \
    peft>=0.4.0 \
    fastapi \
    uvicorn \
    python-multipart \
    pydantic \
    protobuf \
    sentencepiece \
    flask \
    flask-cors \
    pyyaml


    
# Copy your project files
# COPY . /IMGENIE/

# Set working directory
# WORKDIR /IMGENIE

# Default command
CMD ["/bin/bash"]
