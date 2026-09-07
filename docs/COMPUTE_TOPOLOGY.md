# Compute & Hardware Topology Specification

## Cluster Overview & Storage Quotas
- Usable Scratch Storage: 250 GB per node max (preserve operating system buffer).
- Target Model Size Budget: Checkpoints < 2 GB; Hugging Face cache redirected to scratch.

---

## Node 1: Primary Linux Workstation (`node-victus-fedora`)
- Hardware: HP Victus 15
- CPU: AMD Ryzen 7 7445HS (8C/16T)
- GPU: NVIDIA GeForce RTX 3050 Mobile (6 GB GDDR6 VRAM)
- OS / DE: Fedora Linux (x86_64) + GNOME
- Network Constraints: Cloudflare WARP route (requires explicit pip trust / cert bypass flags)
- Primary Assignment:
  - Phase 1 Normalization & Gazetteer parsing
  - Phase 2 Dawid-Skene weak supervision (CPU/NumPy)
  - Synthetic log generation via GGUF/llama.cpp (Q4_K_M)
  - Final target CPU ONNX INT8 deployment benchmarking (4 vCPU thread constraint)
- Shell Protocol: Native Bash

---

## Node 2: Dedicated Fine-Tuning Node (`node-tuf-windows`)
- Hardware: ASUS TUF Gaming A15
- CPU: AMD Ryzen 7 7445HS (8C/16T)
- GPU: NVIDIA GeForce RTX 4050 Mobile (6 GB GDDR6 VRAM, higher TDP)
- OS: Windows 11
- Primary Assignment:
  - Multi-Task DeBERTa-v3 model training (PyTorch + CUDA 12.x)
  - Mixed Precision (FP16 / AMP) gradient descent
  - PyTorch to ONNX graph tracing and float32 model export
- Shell Protocol: PowerShell / Windows Terminal

---

## Execution Constraints by Node

### `node-victus-fedora` (Bash)
```bash
export HF_HOME="/home/hkashyapx/Projects/RIskForge/data/.cache/huggingface"
export TORCH_HOME="/home/hkashyapx/Projects/RIskForge/data/.cache/torch"
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

### `node-tuf-windows` (PowerShell)
```powershell
$env:HF_HOME = "C:\RiskForge\data\.cache\huggingface"
$env:TORCH_HOME = "C:\RiskForge\data\.cache\torch"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pydantic pyyaml onnx onnxruntime
```
