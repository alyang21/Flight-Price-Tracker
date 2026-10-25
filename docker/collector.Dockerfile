# Image for the daily price collector. Lightweight: no pandas or PyTorch.

# 1. Start from an official slim Python image (Debian Linux + Python 3.12).
FROM python:3.12-slim

# 2. All following commands run inside /app in the container.
WORKDIR /app

# 3. Install dependencies BEFORE copying the code. Docker caches each step,
#    so editing collector.py later doesn't force a reinstall of packages.
COPY requirements-collector.txt .
RUN pip install --no-cache-dir -r requirements-collector.txt

# 4. Copy in the code.
COPY collector.py .

# 5. The command the container runs when it starts.
CMD ["python", "collector.py"]
