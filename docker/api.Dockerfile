# Image for the inference API: serves predictions from the trained model.

FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# api.py reuses the feature code in model.py, which imports the other two.
COPY dataset.py baseline.py model.py api.py ./

# Documents the port the server listens on.
EXPOSE 8000

# 0.0.0.0 makes the server reachable from outside the container.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
