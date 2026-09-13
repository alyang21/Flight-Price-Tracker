# Flight Price Tracker & Predictor

An end-to-end system that collects flight price data over time and predicts future price movement, built to demonstrate cloud infrastructure (AWS), containerization (Docker, Kubernetes), and deep learning (PyTorch/TensorFlow) working together as a real pipeline, not just as isolated scripts.

## Project Status

Started: September 2026

- [ ] Phase 1: Data collector + baseline model (local)
- [ ] Phase 2: Containerization (Docker)
- [ ] Phase 3: Cloud deployment (AWS)
- [ ] Phase 4 (stretch): Kubernetes orchestration

## Why This Project

This project  builds its own growing dataset over time by collecting real flight price data on a schedule, then uses that data to train a model that predicts whether prices on a given route are likely to rise or fall. The point of the project is the full pipeline, automated collection, containerized services, cloud deployment, and a deep learning model working together, not just the model in isolation.

## Planned Architecture

```
[Scheduled Data Collector] --> [S3: raw price data]
                                       |
                                       v
                              [Training Job: PyTorch/TensorFlow]
                                       |
                                       v
                              [Trained Model: S3]
                                       |
                                       v
                        [Inference API] <-- [User/request]
```

All components containerized with Docker. Deployed on AWS (S3, EventBridge/Lambda or ECS for scheduling, API Gateway or ECS for serving predictions). Kubernetes orchestration is a stretch goal once the core pipeline is stable.

## Phase Breakdown

### Phase 1: Data Collector + Baseline Model (local only)
- Pull flight price data on a schedule from a flight pricing API for a handful of routes
- Store collected data locally as the dataset grows
- Build a baseline time series model (starting simple, then a PyTorch/TensorFlow model) to predict price direction
- Everything runs on my machine, no cloud or containers yet

### Phase 2: Containerization
- Split the system into separate services: data collector, model training, inference API
- Write a Dockerfile for each service
- Connect them locally with docker-compose
- Confirm the full pipeline runs end to end in containers before touching the cloud

### Phase 3: AWS Deployment
- S3 for storing raw data and trained models
- A scheduled trigger (EventBridge + Lambda, or a container on ECS) running the data collector automatically
- An inference endpoint (API Gateway + Lambda, or ECS service) serving predictions
- IAM roles scoped correctly for each service

### Phase 4 (Stretch): Kubernetes
- Migrate the containerized services to run under Kubernetes instead of directly on ECS
- Primarily a learning exercise in orchestration, not a requirement for this project's actual scale

## Tech Stack (planned)

Python, PyTorch or TensorFlow, Docker, Docker Compose, AWS (S3, Lambda, EventBridge, API Gateway/ECS), Kubernetes (stretch), a flight pricing API (Amadeus or similar)
