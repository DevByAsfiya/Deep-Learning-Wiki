# Implementations

Code that runs. Every entry pairs with a topic note, because an implementation
without an explanation is a snippet, and an explanation without an
implementation is a lecture.

**Template:** [IMPLEMENTATION_TEMPLATE.md](../_templates/IMPLEMENTATION_TEMPLATE.md)

---

## Index

- [From scratch](#from-scratch)
- [Notebooks](#notebooks)
- [Planned](#planned)
- [Conventions](#conventions)

---

## From scratch

<a id="from-scratch"></a>

NumPy only, no framework. The point is to understand the mechanism, not to be
fast.

| Implementation | Explains | Status |
| --- | --- | --- |
| | _Nothing here yet_ | |

## Notebooks

<a id="notebooks"></a>

Interactive experiments, usually paired with a topic note.

| Notebook | Pairs with | Status |
| --- | --- | --- |
| | _Nothing here yet_ | |

## Planned

<a id="planned"></a>

| Implementation | Why it is worth building |
| --- | --- |
| Logistic regression in NumPy | Forward pass, loss, and gradient with nothing hidden |
| Backpropagation from scratch | The one exercise that makes Chapter 1 permanent |
| LeNet-5 on MNIST | First working CNN, trains on a laptop CPU |
| Plain net vs ResNet on CIFAR-10 | Watch the degradation problem happen, then watch skip connections fix it |
| Character level transformer | Attention, positional encoding, and masking in one small file |

## Conventions

<a id="conventions"></a>

- Every implementation has a `README.md` following the template
- Pin dependencies in a `requirements.txt` per implementation
- Include expected output, so a reader knows whether their run is correct
- Keep runtimes short. If it needs more than a few minutes on a laptop, provide
  a smaller configuration as the default
- Record what went wrong during development. Those notes age better than the code
