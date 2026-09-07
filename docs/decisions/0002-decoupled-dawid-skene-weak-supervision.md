# ADR-0002: Pure NumPy Dawid-Skene Engine for Weak Supervision

## Status
Accepted

## Context
Snorkel's LabelModel pulls PyTorch and large dependencies into preprocessing worker nodes. Offline labeling must execute on lightweight ETL containers with memory limits under 500 MB RAM.

## Decision
Implement a pure NumPy Dawid-Skene Expectation-Maximization engine that solves for latent LF error rates and outputs class probabilities without PyTorch dependencies.

## Consequences
- ETL preprocessing runs without GPU or PyTorch runtime packages.
- Offline labeling pipeline is fully decoupled from model training pipeline.
