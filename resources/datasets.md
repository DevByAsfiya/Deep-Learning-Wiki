# Datasets

## For learning

| Dataset | Task | Size | Why it is useful |
| --- | --- | --- | --- |
| MNIST | Handwritten digit classification | 70k images, 28x28 grayscale | Trains in minutes on a CPU. Your first CNN goes here |
| Fashion-MNIST | Clothing classification | 70k images, 28x28 | Drop in replacement for MNIST, meaningfully harder |
| CIFAR-10 | Object classification | 60k images, 32x32 colour | The standard for architecture experiments. Small enough to iterate |
| IMDB reviews | Sentiment classification | 50k reviews | First text classification task |
| Tiny Shakespeare | Character level language modelling | ~1MB text | Small enough to train a transformer from scratch on a laptop |

## For benchmarking

| Dataset | Task | Notes |
| --- | --- | --- |
| ImageNet | 1000 class classification | The historical benchmark. Pretrained weights everywhere |
| COCO | Detection and segmentation | The standard for detection |
| GLUE, SuperGLUE | Language understanding | Largely saturated, but still cited |

## Choosing a dataset to learn on

Pick the smallest dataset that still exhibits the phenomenon you want to study.
If you are learning about overfitting, a dataset that takes six hours per epoch
teaches you nothing per unit of time. Iteration speed matters more than realism
while you are learning.

Once you are building something real, the opposite applies: your own data,
however messy, teaches you more than any benchmark.
