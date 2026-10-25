# Image for the training job: loads the collected data, trains the model,
# and saves it to the models folder.

FROM python:3.12-slim
WORKDIR /app

# Install the CPU-only build of PyTorch. The default build includes NVIDIA GPU
# libraries (several GB) that this small model never uses.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# The remaining dependencies.
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Only the code training needs; the collector is not included.
COPY dataset.py baseline.py model.py ./

CMD ["python", "model.py"]
