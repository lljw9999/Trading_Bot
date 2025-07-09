# Triton Inference Server with HuggingFace Model Support
FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install HuggingFace Hub CLI and dependencies
ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

RUN pip install --no-cache-dir \
    huggingface-hub>=0.20.0 \
    transformers>=4.36.0 \
    torch>=2.0.0 \
    onnx>=1.14.0 \
    onnxruntime>=1.16.0 \
    numpy>=1.24.0

# Login to HuggingFace Hub if token provided
RUN if [ -n "${HF_TOKEN}" ]; then \
        huggingface-cli login --token ${HF_TOKEN} --add-to-git-credential; \
    fi

# Copy model repository
COPY model_repository /models

# Set model repository path
ENV TRITON_MODEL_REPOSITORY=/models

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton server
CMD ["tritonserver", "--model-repository=/models", "--allow-http=true", "--allow-grpc=true", "--allow-metrics=true"] 