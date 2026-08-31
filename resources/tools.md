# Tools

## Core

| Tool | Purpose | Notes |
| --- | --- | --- |
| **PyTorch** | The framework | Learn this first. Eager by default, so debugging is normal Python debugging |
| **timm** | Pretrained vision backbones | Roughly a thousand models, one API. What practitioners actually use |
| **Hugging Face Transformers** | Pretrained language models | The standard layer above the frameworks |
| **NumPy** | Array computing | Understand broadcasting properly and half your shape bugs disappear |

## Training and experiment tracking

| Tool | Purpose |
| --- | --- |
| **Weights and Biases** or **MLflow** | Experiment tracking. Use one from day one, memory is not a log |
| **PyTorch Lightning** | Removes training loop boilerplate. Learn the raw loop first |
| **Optuna** | Hyperparameter search |

## Data

| Tool | Purpose |
| --- | --- |
| **Albumentations** | Image augmentation, faster and richer than torchvision transforms |
| **NVIDIA DALI** | GPU side data loading, when augmentation becomes the bottleneck |
| **DVC** | Version control for datasets |

## Deployment

| Tool | Purpose |
| --- | --- |
| **ONNX** | Framework independent model format |
| **TensorRT** | NVIDIA inference optimization |
| **vLLM** | High throughput LLM serving |
| **BentoML** or **TorchServe** | Model serving |

## A warning about tooling

It is easy to spend a month evaluating tools and zero weeks training models.
Default to PyTorch plus one experiment tracker, and add a tool only when a
specific problem forces it.
