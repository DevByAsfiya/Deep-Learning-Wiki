# Computer Vision

Convolutional networks, detection, segmentation, and vision transformers.

**Prerequisites:** [01 Foundations](../01-foundations/INDEX.md),
[02 Training and Optimization](../02-training-and-optimization/INDEX.md)

---

## Contents

| Document | Covers | Level | Status |
| --- | --- | --- | --- |
| [CNN Classic Architectures](architectures/cnn-classic-architectures.md) | LeNet-5, AlexNet, VGG-16, ResNets, 1x1 convolutions, Inception, transfer learning, data augmentation | beginner | Complete |

### Subfolders

| Subfolder | Covers | Status |
| --- | --- | --- |
| [`fundamentals/`](fundamentals/) | Convolution, padding, stride, pooling | Stub |
| [`architectures/`](architectures/) | Named model architectures and what each contributed | 1 document |
| [`object-detection/`](object-detection/) | R-CNN family, YOLO, DETR | Stub |
| [`segmentation/`](segmentation/) | Semantic, instance, and panoptic segmentation | Stub |
| [`vision-transformers/`](vision-transformers/) | Attention applied to images | Stub |
| [`applications/`](applications/) | Vision systems in production | Stub |

---

## The through-line of the architectures document

The five architectures are not a list, they are an argument. Each one responds
to a problem the previous one created:

```text
LeNet-5     establishes CONV -> POOL -> FC
   |
AlexNet     proves depth works, given ReLU, dropout, and GPUs
   |
VGG-16      makes it uniform: 3x3 everywhere, repeated blocks
   |
   |  problem: deeper networks now get WORSE training error
   |
ResNet      skip connections make the identity easy to learn
   |
   |  problem: depth is expensive
   |
1x1 conv    channel count becomes cheap to control
Inception   parallel multi-scale, made affordable by bottlenecks
```

| If you want | Go to |
| --- | --- |
| Why deeper networks got worse | [The problem ResNets solve](architectures/cnn-classic-architectures.md#4-the-problem-resnets-solve) |
| The residual block derivation | [The residual block](architectures/cnn-classic-architectures.md#5-the-residual-block) |
| What a 1x1 convolution actually does | [1x1 convolutions](architectures/cnn-classic-architectures.md#6-what-does-a-11-convolution-do) |
| The 120M to 12.4M cost calculation | [The bottleneck fix](architectures/cnn-classic-architectures.md#8-the-computational-cost-problem-and-the-bottleneck-fix) |
| How to fine tune a pretrained model | [Transfer learning](architectures/cnn-classic-architectures.md#11-transfer-learning) |
| What to build, in what order | [Implementation roadmap](architectures/cnn-classic-architectures.md#part-6-your-implementation-roadmap) |

---

## Status legend

| Status | Meaning |
| --- | --- |
| Complete | Written, reviewed, and safe to learn from |
| Draft | Written but not yet reviewed |
| Stub | Placeholder, contributions welcome |

---

## Related sections

- [01 Foundations](../01-foundations/INDEX.md)
- [09 Efficiency and Deployment](../09-efficiency-and-deployment/INDEX.md), where the 1x1 convolution ideas lead
- [Master index](../INDEX.md)
