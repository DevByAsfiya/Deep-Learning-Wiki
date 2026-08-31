# Prerequisites

> **Level:** beginner
> **Prerequisites:** none

What you need before starting, and where to get it. This list is deliberately
short. Most people over prepare on mathematics and under prepare on code.

---

## Index

- [Programming](#programming)
- [Mathematics](#mathematics)
- [Tooling](#tooling)
- [What you do not need yet](#what-you-do-not-need-yet)
- [Self check](#self-check)

---

## Programming

<a id="programming"></a>

| Skill | Why it matters |
| --- | --- |
| Python basics: functions, classes, loops, list comprehensions | Everything here is Python |
| NumPy arrays: indexing, slicing, broadcasting | Tensor operations are NumPy with a GPU |
| Reading a stack trace | Most of your time will be spent on shape errors |

Broadcasting in particular is worth an hour of focused practice. The single most
common deep learning bug is a shape mismatch that silently broadcasts instead of
raising an error.

## Mathematics

<a id="mathematics"></a>

| Topic | Depth needed |
| --- | --- |
| **Linear algebra** | What a vector and matrix are, matrix multiplication, transpose, and what a shape means |
| **Calculus** | What a derivative measures, the chain rule. You will never compute one by hand in practice |
| **Probability** | Probability distributions, mean and variance, what a log probability is |

You do not need proofs, eigendecompositions, or measure theory to start. Pick up
depth when a specific topic demands it, and do not let mathematics anxiety delay
writing your first training loop.

## Tooling

<a id="tooling"></a>

```bash
# a working environment
python -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib jupyter
```

| Tool | Purpose |
| --- | --- |
| **PyTorch** | The framework used throughout this repository |
| **Jupyter or Colab** | Interactive experimentation. Colab gives free GPU access |
| **Git** | For this repository, and for every project you build |

Learn PyTorch rather than TensorFlow first. The concepts transfer completely,
and PyTorch is easier to debug because it runs eagerly by default.

## What you do not need yet

<a id="what-you-do-not-need-yet"></a>

- A GPU. Everything in `01` and `02` runs on a laptop CPU.
- A maths degree.
- To have finished a course. Learn by building, and read to unblock yourself.

## Self check

<a id="self-check"></a>

If you can answer these, you are ready:

1. What does `x[:, None] * y` do when `x` has shape `(5,)` and `y` has shape `(3,)`?
2. If `A` is `(4, 3)` and `B` is `(3, 7)`, what shape is `A @ B`?
3. What does it mean, in words, that the derivative of a function is negative at a point?

If any of those were shaky, spend a few hours there first. It will pay back
immediately.
