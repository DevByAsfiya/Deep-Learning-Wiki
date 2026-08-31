# Learning Paths

> **Level:** everyone
> **Prerequisites:** [prerequisites.md](prerequisites.md)

Different goals need different routes. Pick the one that matches yours rather
than reading everything in order.

---

## Index

- [Path 1: Complete beginner](#path-1-complete-beginner)
- [Path 2: Computer vision](#path-2-computer-vision)
- [Path 3: NLP and language models](#path-3-nlp-and-language-models)
- [Path 4: Shipping a model](#path-4-shipping-a-model)
- [Path 5: Reading research](#path-5-reading-research)
- [How long each takes](#how-long-each-takes)

---

## Path 1: Complete beginner

<a id="path-1-complete-beginner"></a>

You have never trained a model. Goal: understand what a neural network is and
train one yourself.

1. [Prerequisites](prerequisites.md), and fix any gaps
2. [Deep Learning Fundamentals](../01-foundations/neural-networks/deep-learning-fundamentals.md),
   Chapter 1 only. Stop at the end of section 1.3
3. **Build:** logistic regression in NumPy, no framework. Then the same thing in PyTorch
4. Fundamentals, Chapter 2. Optimization, regularization, and tuning
5. **Build:** an MLP on a tabular dataset. Deliberately overfit it, then fix it with regularization
6. Fundamentals, Chapter 4. ML strategy and error analysis

Do not skip step 3. Implementing backpropagation once, by hand, is worth more
than reading about it ten times.

## Path 2: Computer vision

<a id="path-2-computer-vision"></a>

You want to work with images.

1. Complete Path 1 through step 4
2. [Computer Vision](../03-computer-vision/INDEX.md) fundamentals: convolution, padding, stride, pooling
3. [CNN Classic Architectures](../03-computer-vision/architectures/cnn-classic-architectures.md)
4. **Build:** LeNet-5 on MNIST, then a small ResNet on CIFAR-10
5. **Run the key experiment:** train a 20 layer plain network and a 20 layer ResNet, plot both training losses, and watch the plain one plateau higher
6. **Build:** fine tune a pretrained ResNet-50 on a few hundred of your own images
7. Object detection, then segmentation

## Path 3: NLP and language models

<a id="path-3-nlp-and-language-models"></a>

1. Complete Path 1 through step 4
2. [Sequence Models](../05-sequence-models/INDEX.md): why RNNs exist and why they fail
3. Attention, then the transformer architecture
4. **Build:** a character level transformer from scratch. Small, on a small corpus
5. Pretraining objectives and model families
6. Fine tuning: LoRA and adapters
7. Retrieval augmented generation, then agents

## Path 4: Shipping a model

<a id="path-4-shipping-a-model"></a>

You can train models and now need them in production.

1. [ML strategy and error analysis](../01-foundations/neural-networks/deep-learning-fundamentals.md#chapter-4-structuring-machine-learning-projects)
2. [Efficiency and Deployment](../09-efficiency-and-deployment/INDEX.md): quantization, distillation, pruning
3. Serving: batching, latency budgets, hardware choices
4. Monitoring: drift detection and retraining triggers
5. [Responsible AI](../10-responsible-ai/INDEX.md): evaluation and robustness

**Measure latency before you optimise anything.** Most teams optimise the wrong
component because they never profiled.

## Path 5: Reading research

<a id="path-5-reading-research"></a>

1. Read the [paper notes index](../papers/INDEX.md) for papers already summarised
2. Read papers in **historical order** within a topic. Each one is usually a
   direct response to the previous one's limitation, and that narrative is the
   fastest way to understand a subfield
3. Write a note for every paper using the
   [template](../_templates/PAPER_NOTE_TEMPLATE.md). If you cannot write the one
   line contribution, read it again
4. Reimplement one paper per quarter. Reading is not understanding

## How long each takes

<a id="how-long-each-takes"></a>

| Path | Realistic time at ten hours per week |
| --- | --- |
| 1. Complete beginner | 6 to 8 weeks |
| 2. Computer vision | 6 weeks after Path 1 |
| 3. NLP | 8 weeks after Path 1 |
| 4. Shipping | 4 weeks, alongside a real project |
| 5. Research | continuous |

These assume you build at every step. Reading only, the numbers are shorter and
the retention is much worse.
