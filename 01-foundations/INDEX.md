# Foundations

The mathematics and core neural network ideas everything else is built on. If
you read one section of this repository properly, make it this one. Every later
section assumes the vocabulary defined here.

**Prerequisites:** school level algebra and basic Python. See
[prerequisites](../00-start-here/prerequisites.md).

---

## Contents

| Document | Covers | Level | Status |
| --- | --- | --- | --- |
| [Deep Learning Fundamentals](neural-networks/deep-learning-fundamentals.md) | Neurons through deep networks, optimization, regularization, batch norm, softmax, ML strategy, error analysis, transfer and multi-task learning, end-to-end deep learning | beginner | Complete |

### Subfolders

| Subfolder | Covers | Status |
| --- | --- | --- |
| [`mathematics/`](mathematics/) | Linear algebra, calculus, and probability, only the parts you actually need | Stub |
| [`machine-learning-basics/`](machine-learning-basics/) | Supervised learning, loss functions, evaluation, before any deep learning | Stub |
| [`neural-networks/`](neural-networks/) | Neurons, layers, forward and backward propagation, deep networks | 1 document |

---

## What is inside the fundamentals document

It is long, so here are the entry points:

| If you want | Go to |
| --- | --- |
| What a neural network is | [1.1.1](neural-networks/deep-learning-fundamentals.md#111-what-a-neural-network-actually-is) |
| Notation used across the whole field | [1.1.5](neural-networks/deep-learning-fundamentals.md#115-notation) |
| Backpropagation | [1.1.9](neural-networks/deep-learning-fundamentals.md#119-computation-graphs-and-backpropagation) |
| Which activation function to use | [1.2.3](neural-networks/deep-learning-fundamentals.md#123-activation-functions) |
| Bias vs variance, and what to do about each | [2.1.3](neural-networks/deep-learning-fundamentals.md#213-bias-and-variance) |
| Adam and the other optimizers | [2.2.11](neural-networks/deep-learning-fundamentals.md#2211-adam-optimization) |
| Batch normalization | [2.3.6](neural-networks/deep-learning-fundamentals.md#236-batch-normalization-normalizing-activations-in-a-network) |
| Softmax and multi-class classification | [2.3.14](neural-networks/deep-learning-fundamentals.md#2314-multi-class-classification-and-softmax-regression) |
| How to prioritise work on a project | [4.1.2 Orthogonalization](neural-networks/deep-learning-fundamentals.md#412-orthogonalization) |
| Transfer learning | [3.1](neural-networks/deep-learning-fundamentals.md#31-transfer-learning) |
| Every term defined | [Appendix A](neural-networks/deep-learning-fundamentals.md#appendix-a-complete-jargon-glossary) |
| Decision cheat sheets | [Appendix B](neural-networks/deep-learning-fundamentals.md#appendix-b-decision-cheat-sheets) |

---

## Status legend

| Status | Meaning |
| --- | --- |
| Complete | Written, reviewed, and safe to learn from |
| Draft | Written but not yet reviewed |
| Stub | Placeholder, contributions welcome |

---

## Related sections

- [02 Training and Optimization](../02-training-and-optimization/INDEX.md), which goes deeper on everything in Chapter 2
- [03 Computer Vision](../03-computer-vision/INDEX.md), the natural next step
- [Master index](../INDEX.md)
