# Deep Learning for Computer Vision: CNN Architectures (Part II)

> **Source:** lecture slides, expanded into a full guide
> **Level:** beginner to intermediate
> **Prerequisites:** convolution, padding, stride, pooling, fully connected layers
> **Covers:** LeNet-5, AlexNet, VGG-16, ResNets, 1x1 convolutions, Inception/GoogLeNet, transfer learning, data augmentation, the state of computer vision
> **Related:** [Deep Learning Fundamentals](../../01-foundations/neural-networks/deep-learning-fundamentals.md)

Every topic below is laid out the same way: the plain English version, the maths,
the intuition, runnable code, and where it shows up in production systems.

---

## Index

- **[How to read this](#how-to-read-this)**
- **[Part 0, Why case studies at all?](#part-0-why-case-studies-at-all)**

### [PART 1, CLASSIC NETWORKS](#part-1-classic-networks)

- **[1. LeNet-5 (1998)](#1-lenet-5-1998)**
  - [Plain English](#plain-english)
  - [The architecture, layer by layer](#the-architecture-layer-by-layer)
  - [The maths](#the-maths)
  - [The intuition, the pattern to memorise](#the-intuition-the-pattern-to-memorise)
  - [Historical notes (things that changed since)](#historical-notes-things-that-changed-since)
  - [Implementation](#implementation)
  - [In industry](#in-industry)
- **[2. AlexNet (2012)](#2-alexnet-2012)**
  - [Plain English](#plain-english-1)
  - [The architecture](#the-architecture)
  - [The maths](#the-maths-1)
  - [The intuition, what actually made it work](#the-intuition-what-actually-made-it-work)
  - [Implementation](#implementation-1)
  - [In industry](#in-industry-1)
- **[3. VGG-16 (2015)](#3-vgg-16-2015)**
  - [Plain English](#plain-english-2)
  - [The architecture](#the-architecture-1)
  - [The maths](#the-maths-2)
  - [The intuition](#the-intuition)
  - [Implementation](#implementation-2)
  - [In industry](#in-industry-2)

### [PART 2, RESIDUAL NETWORKS (ResNets)](#part-2-residual-networks-resnets)

- **[4. The problem ResNets solve](#4-the-problem-resnets-solve)**
  - [Plain English](#plain-english-3)
  - [The plain vs residual curve](#the-plain-vs-residual-curve)
- **[5. The residual block](#5-the-residual-block)**
  - [The maths, a plain block](#the-maths-a-plain-block)
  - [The maths, a residual block](#the-maths-a-residual-block)
  - [Why it works, the identity argument](#why-it-works-the-identity-argument)
  - [The gradient view (why gradients stop vanishing)](#the-gradient-view-why-gradients-stop-vanishing)
  - [The dimension problem, and where 1×1 convs first appear](#the-dimension-problem-and-where-11-convs-first-appear)
  - [ResNet-34 and the bottleneck (ResNet-50+)](#resnet-34-and-the-bottleneck-resnet-50)
  - [Implementation](#implementation-3)
  - [In industry](#in-industry-3)

### [PART 3, 1×1 CONVOLUTIONS (Network in Network)](#part-3-11-convolutions-network-in-network)

- **[6. What does a 1×1 convolution do?](#6-what-does-a-11-convolution-do)**
  - [Plain English](#plain-english-4)
  - [The maths](#the-maths-3)
  - [The intuition, three distinct uses](#the-intuition-three-distinct-uses)
  - [The mental model that makes it click](#the-mental-model-that-makes-it-click)
  - [Implementation](#implementation-4)
  - [In industry](#in-industry-4)

### [PART 4, INCEPTION NETWORKS](#part-4-inception-networks)

- **[7. Inception motivation](#7-inception-motivation)**
  - [Plain English](#plain-english-5)
  - [The module (from your slide)](#the-module-from-your-slide)
  - [The intuition](#the-intuition-1)
- **[8. The computational cost problem, and the bottleneck fix](#8-the-computational-cost-problem-and-the-bottleneck-fix)**
  - [The expensive version](#the-expensive-version)
  - [The bottleneck version](#the-bottleneck-version)
  - [Does shrinking to 16 channels destroy information?](#does-shrinking-to-16-channels-destroy-information)
  - [The full Inception module with bottlenecks](#the-full-inception-module-with-bottlenecks)
- **[9. The Inception network (GoogLeNet)](#9-the-inception-network-googlenet)**
  - [Plain English](#plain-english-6)
  - [Auxiliary classifiers](#auxiliary-classifiers)
  - [And the name](#and-the-name)
  - [Implementation](#implementation-5)
  - [In industry](#in-industry-5)

### [PART 5, PRACTICAL ADVICE FOR USING CONVNETS](#part-5-practical-advice-for-using-convnets)

- **[10. Use open source implementations](#10-use-open-source-implementations)**
  - [The advice](#the-advice)
  - [Why this is not laziness](#why-this-is-not-laziness)
  - [How to actually do it](#how-to-actually-do-it)
  - [In industry](#in-industry-6)
- **[11. Transfer learning](#11-transfer-learning)**
  - [Plain English](#plain-english-7)
  - [The three regimes (your slide shows all three)](#the-three-regimes-your-slide-shows-all-three)
  - [The rule of thumb](#the-rule-of-thumb)
  - [Implementation](#implementation-6)
  - [In industry](#in-industry-7)
- **[12. Data augmentation](#12-data-augmentation)**
  - [Plain English](#plain-english-8)
  - [Common methods (from your slides)](#common-methods-from-your-slides)
  - [Critical: augmentations must preserve the label](#critical-augmentations-must-preserve-the-label)
  - [Implementation](#implementation-7)
  - [Modern augmentations worth knowing](#modern-augmentations-worth-knowing)
  - [In industry](#in-industry-8)
- **[13. The state of computer vision](#13-the-state-of-computer-vision)**
  - [Data vs hand engineering](#data-vs-hand-engineering)
  - [Tips for benchmarks and competitions](#tips-for-benchmarks-and-competitions)
  - [The honest caveat](#the-honest-caveat)

### [PART 6, YOUR IMPLEMENTATION ROADMAP](#part-6-your-implementation-roadmap)

  - [Week 1, Get the shapes in your fingers](#week-1-get-the-shapes-in-your-fingers)
  - [Week 2, Blocks, not layers](#week-2-blocks-not-layers)
  - [Week 3, Efficiency](#week-3-efficiency)
  - [Week 4, The actual job](#week-4-the-actual-job)
- **[The five ideas, if you remember nothing else](#the-five-ideas-if-you-remember-nothing-else)**
- **[Papers, in reading order](#papers-in-reading-order)**

---

---

## How to read this

<a id="how-to-read-this"></a>

Every topic below is laid out the same way:

1. **The plain-English version**: what problem exists, what the idea is
2. **The maths**: the actual equations and number-crunching
3. **The intuition**: *why* it works, the thing that makes it click
4. **Implementation**: code you can actually run
5. **In industry**: where this shows up in real production systems

Work through it in order. Each topic is a building block for the next.

---

## Part 0, Why case studies at all?

<a id="part-0-why-case-studies-at-all"></a>

You already know the pieces: convolution, padding, stride, pooling, fully-connected layers. That's like knowing what bricks, cement and steel are.

Case studies teach you **architecture**: how to stack those pieces so the thing actually stands up. And here's the practical truth: an architecture that works well on one vision task almost always works well on another. The people who won ImageNet spent years finding good stacking patterns. You get to just copy them.

So the goal of this whole lecture is:
- Read a CNN architecture diagram fluently
- Understand the 3–4 big ideas that made deep networks trainable (residuals, 1×1 convs, inception)
- Know the practical playbook (transfer learning, augmentation) so you never train from scratch unnecessarily

The through-line across all of it: **networks kept getting deeper, and each architecture is a trick for making "deeper" actually work.**

---

# PART 1, CLASSIC NETWORKS

<a id="part-1-classic-networks"></a>

---

## 1. LeNet-5 (1998)

<a id="1-lenet-5-1998"></a>

### Plain English

<a id="plain-english"></a>

The first CNN that genuinely worked in production. Yann LeCun built it to read handwritten digits on cheques for banks. It's tiny by modern standards, about **60,000 parameters**: but the shape of it is the shape of every CNN since.

The pattern it established:

```
CONV → POOL → CONV → POOL → FC → FC → OUTPUT
```

### The architecture, layer by layer

<a id="the-architecture-layer-by-layer"></a>

Input: a **32×32×1** grayscale image (one channel, no colour).

| Step | Operation | Output shape |
|---|---|---|
| Input |, | 32×32×1 |
| CONV1 | 6 filters, 5×5, stride 1 | 28×28×6 |
| POOL1 | avg pool, f=2, s=2 | 14×14×6 |
| CONV2 | 16 filters, 5×5, stride 1 | 10×10×16 |
| POOL2 | avg pool, f=2, s=2 | 5×5×16 = 400 |
| FC3 | 400 → 120 | 120 |
| FC4 | 120 → 84 | 84 |
| Output | 84 → 10, softmax | 10 digits |

### The maths

<a id="the-maths"></a>

The output size of a conv layer:

$$n_{out} = \left\lfloor \frac{n_{in} + 2p - f}{s} \right\rfloor + 1$$

Check the first layer: $n_{in}=32$, padding $p=0$ ("valid"), filter $f=5$, stride $s=1$:

$$n_{out} = \frac{32 + 0 - 5}{1} + 1 = 28 \quad \checkmark$$

Pooling with $f=2, s=2$ halves it: $\frac{28-2}{2}+1 = 14$ ✓

**Parameter count for a conv layer:**

$$\text{params} = (f \times f \times n_c^{prev} + 1) \times n_c^{next}$$

CONV1: $(5 \times 5 \times 1 + 1) \times 6 = 156$ parameters.
CONV2: $(5 \times 5 \times 6 + 1) \times 16 = 2{,}416$ parameters.

Notice how small those are. The magic of convolution is **parameter sharing**: the same 5×5 filter slides over the whole image, so you learn one edge detector, not one per pixel location.

### The intuition, the pattern to memorise

<a id="the-intuition-the-pattern-to-memorise"></a>

Look at the shapes as you go deeper:

$$n_H, n_W \downarrow \qquad n_c \uparrow$$

**Height and width shrink. Channel depth grows.**

32×32×1 → 28×28×6 → 14×14×6 → 10×10×16 → 5×5×16

This is the single most important structural idea in CNNs. Early layers have high spatial resolution but few feature types (edges, blobs). Deep layers have low spatial resolution but many feature types (wheels, faces, textures). You're trading *where* for *what*.

### Historical notes (things that changed since)

<a id="historical-notes-things-that-changed-since"></a>

- LeNet used **sigmoid/tanh**, not ReLU. ReLU didn't exist in mainstream use yet.
- It applied the **non-linearity after pooling**, which is unusual today.
- It used **average pooling**; max pooling now dominates.
- It had no padding, so every conv shrank the image.

Don't copy these. Copy the *shape*, not the details.

### Implementation

<a id="implementation"></a>

```python
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),   # 32x32x1 -> 28x28x6
            nn.ReLU(),                         # modernised: ReLU not tanh
            nn.AvgPool2d(2, 2),                # -> 14x14x6
            nn.Conv2d(6, 16, kernel_size=5),   # -> 10x10x16
            nn.ReLU(),
            nn.AvgPool2d(2, 2),                # -> 5x5x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # -> 400
            nn.Linear(400, 120), nn.ReLU(),
            nn.Linear(120, 84),  nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

Train it on MNIST. It takes about two minutes on a laptop CPU and hits ~99% accuracy. **This is your first exercise, do it today.**

### In industry

<a id="in-industry"></a>

LeNet-scale networks are still everywhere in **embedded and edge vision**, because they're small enough to run on a microcontroller:

- **Cheque and document processing**: banks still run digit/character recognisers on scanned fields. The modern version is a small CNN feeding into an OCR pipeline.
- **Industrial defect detection on production lines**: a camera over a conveyor, classifying "pass/fail" on a small crop. A 60K-parameter net runs at 1000fps on a cheap ARM chip.
- **Meter reading, licence plate character classification, barcode fallback**: anything where the input is a small, clean, cropped patch.

The lesson: when your input is small and the task is narrow, **don't reach for ResNet-50**. A LeNet-scale net is faster, cheaper, and often just as accurate.

---

## 2. AlexNet (2012)

<a id="2-alexnet-2012"></a>

### Plain English

<a id="plain-english-1"></a>

The paper that started the deep learning boom. AlexNet won ImageNet 2012 by such a huge margin (top-5 error dropped from ~26% to ~15%) that the entire field switched to neural networks within about a year.

Structurally it's LeNet, but **~1000× bigger**: 60 million parameters instead of 60 thousand, and trained on 1.2 million images across 1000 classes.

### The architecture

<a id="the-architecture"></a>

Input: **227×227×3** (colour images).

| Step | Operation | Output shape |
|---|---|---|
| Input |, | 227×227×3 |
| CONV1 | 96 filters, 11×11, **s=4** | 55×55×96 |
| MAXPOOL | 3×3, s=2 | 27×27×96 |
| CONV2 | 256 filters, 5×5, same | 27×27×256 |
| MAXPOOL | 3×3, s=2 | 13×13×256 |
| CONV3 | 384 filters, 3×3, same | 13×13×384 |
| CONV4 | 384 filters, 3×3, same | 13×13×384 |
| CONV5 | 256 filters, 3×3, same | 13×13×256 |
| MAXPOOL | 3×3, s=2 | 6×6×256 = **9216** |
| FC6 | 9216 → 4096 | 4096 |
| FC7 | 4096 → 4096 | 4096 |
| Output | 4096 → 1000, softmax | 1000 |

### The maths

<a id="the-maths-1"></a>

Check CONV1 with the formula: $n_{in}=227$, $f=11$, $s=4$, $p=0$:

$$n_{out} = \frac{227 - 11}{4} + 1 = \frac{216}{4} + 1 = 55 \quad \checkmark$$

That stride of 4 is aggressive, it immediately downsamples 4× to keep computation manageable on 2012 GPUs.

**Where the 60M parameters live** (this is important):

| Layer | Parameters |
|---|---|
| All 5 conv layers combined | ~3.7M |
| FC6 (9216 → 4096) | 9216 × 4096 ≈ **37.7M** |
| FC7 (4096 → 4096) | ≈ **16.8M** |
| FC8 (4096 → 1000) | ≈ 4.1M |

**~95% of the parameters are in the three fully-connected layers.** The convolutional layers, the part doing the actual visual work, are only 6%. This is a scandal that later architectures (Inception, ResNet) fixed by removing the giant FC layers entirely.

### The intuition, what actually made it work

<a id="the-intuition-what-actually-made-it-work"></a>

Three things, and they're all still standard practice:

**1. ReLU instead of sigmoid/tanh.**

$$g(z) = \max(0, z)$$

Sigmoid saturates: for large $|z|$, the gradient $\sigma'(z) = \sigma(z)(1-\sigma(z)) \to 0$. Gradients vanish and deep nets stop learning. ReLU has gradient exactly 1 for all $z > 0$, no saturation, no vanishing. AlexNet trained ~6× faster than the tanh equivalent.

**2. Dropout in the FC layers.** With 60M parameters and 1.2M images, the net can memorise. Dropout randomly zeroes 50% of FC activations during training, forcing redundant representations.

**3. Trained on two GPUs** (a 2012 memory constraint that split the network in half). You can ignore this, it's a historical artifact, not a design principle.

### Implementation

<a id="implementation-1"></a>

```python
import torchvision.models as models

# Don't write AlexNet from scratch - it's built in
model = models.alexnet(weights='IMAGENET1K_V1')
print(model)
```

But do write it once by hand to internalise the shape calculations. Then check your output shapes against the table above with a dummy tensor:

```python
import torch
x = torch.randn(1, 3, 227, 227)
for layer in model.features:
    x = layer(x)
    print(type(layer).__name__, tuple(x.shape))
```

### In industry

<a id="in-industry-1"></a>

AlexNet itself is obsolete, nobody deploys it in 2026. But its **legacy is the standard pipeline**:

- **The pretrained-backbone pattern.** AlexNet proved that features learned on ImageNet transfer to totally unrelated tasks. Every production vision system today starts from a pretrained backbone. That idea starts here.
- **The conv-stack-then-classifier-head split.** Modern object detectors (YOLO, Faster R-CNN) are literally "AlexNet-descendant backbone + a different head bolted on".

Where you'll see the actual influence: **feature extraction for visual search**. Take the 4096-dim FC7 activation as an image embedding, index it in a vector database, and you have reverse image search. E-commerce "find similar products" was built this way for years (modern versions use CLIP or a ResNet/ViT embedding, but the architecture of the *idea* is unchanged).

---

## 3. VGG-16 (2015)

<a id="3-vgg-16-2015"></a>

### Plain English

<a id="plain-english-2"></a>

VGG asked a simple question: *what if we stop hand-tuning filter sizes and just use 3×3 everywhere?*

The whole architecture uses exactly two operations:

- **CONV = 3×3 filter, stride 1, "same" padding**
- **MAXPOOL = 2×2, stride 2**

That's it. The only thing that varies is how many filters and how many times you repeat. This uniformity is why VGG became the teaching example, it's beautifully simple. It's also huge: **~138 million parameters**.

### The architecture

<a id="the-architecture-1"></a>

Input: **224×224×3**.

```
224×224×3
  [CONV 64] ×2   → 224×224×64
  POOL           → 112×112×64
  [CONV 128] ×2  → 112×112×128
  POOL           → 56×56×128
  [CONV 256] ×3  → 56×56×256
  POOL           → 28×28×256
  [CONV 512] ×3  → 28×28×512
  POOL           → 14×14×512
  [CONV 512] ×3  → 14×14×512
  POOL           → 7×7×512
  FC 4096
  FC 4096
  Softmax 1000
```

The "16" means 16 layers with learnable weights (13 conv + 3 FC).

### The maths

<a id="the-maths-2"></a>

**Why "same" padding keeps size constant.** For a 3×3 filter with stride 1, set:

$$p = \frac{f-1}{2} = \frac{3-1}{2} = 1$$

Then $n_{out} = \frac{n + 2(1) - 3}{1} + 1 = n$. Size preserved. Every conv layer in VGG preserves spatial size; **only pooling shrinks it**. That's what makes the architecture so easy to reason about.

**The doubling pattern.** Notice at each pooling stage:

$$n_H, n_W \downarrow \times 0.5 \qquad n_c \uparrow \times 2$$

64 → 128 → 256 → 512 (then it caps at 512 because memory). This is a deliberate, systematic version of the LeNet pattern.

**Why 3×3 stacked beats one big filter.** Two stacked 3×3 convs have the same *receptive field* as one 5×5:

- Layer 1: each output sees 3×3 of the input
- Layer 2: each output sees 3×3 of layer 1, which spans 5×5 of the original input

But count the parameters (for $C$ input and output channels):

$$\text{two } 3\times3: \quad 2 \times (3 \times 3 \times C \times C) = 18C^2$$
$$\text{one } 5\times5: \quad 5 \times 5 \times C \times C = 25C^2$$

**28% fewer parameters, plus an extra ReLU in between** (so more non-linearity, more expressive power). Three stacked 3×3s match a 7×7 with $27C^2$ vs $49C^2$, a 45% saving.

This argument is why **almost nothing uses filters bigger than 3×3 anymore.**

### The intuition

<a id="the-intuition"></a>

VGG's real contribution wasn't accuracy, it was proving that **depth with a simple, repeated, uniform block** beats clever hand-designed heterogeneous layers. Every architecture after it (ResNet, Inception, even Vision Transformers) follows the same philosophy: design one good block, repeat it.

Its weakness: **138M parameters and ~15 GFLOPs per image.** It's slow and memory-hungry, mostly because of that 7×7×512 → 4096 FC layer (25088 × 4096 ≈ 103M parameters in a single layer).

### Implementation

<a id="implementation-2"></a>

```python
import torch.nn as nn

def vgg_block(in_c, out_c, n_convs):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, padding=1),
                   nn.ReLU(inplace=True)]
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(3,   64,  2),
            vgg_block(64,  128, 2),
            vgg_block(128, 256, 3),
            vgg_block(256, 512, 3),
            vgg_block(512, 512, 3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096),    nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

Writing `vgg_block` yourself is the exercise. It teaches you to think in **blocks**, not layers, which is how all modern architecture code is structured.

### In industry

<a id="in-industry-2"></a>

VGG is too slow for real-time inference, but it survives in two specific niches where its simple, uniform feature hierarchy is an asset:

- **Perceptual loss / style transfer.** When you train an image generator (super-resolution, denoising, style transfer), you don't compare pixels, you compare *VGG features*. Loss functions like `VGGLoss` or `LPIPS` run both images through a frozen pretrained VGG and compare intermediate activations. This is standard in production super-resolution and image restoration pipelines, and still used to evaluate diffusion models.
- **Teaching and baselines.** When a research paper needs a "standard CNN backbone" for a controlled comparison, VGG-16 is often the reference point because everyone's numbers are comparable.

Practical takeaway for you: **use VGG when you need a well-understood frozen feature extractor. Never use it when latency matters.**

---

# PART 2, RESIDUAL NETWORKS (ResNets)

<a id="part-2-residual-networks-resnets"></a>

---

## 4. The problem ResNets solve

<a id="4-the-problem-resnets-solve"></a>

### Plain English

<a id="plain-english-3"></a>

VGG showed depth helps. So people tried going deeper, 20, 30, 50 layers, and found something bizarre:

**Deeper networks performed *worse*, even on the training set.**

That's not overfitting. Overfitting means great training error, bad test error. This was bad training error. The deep network literally could not fit the data as well as the shallower one.

This is strange, because a 50-layer network *contains* a 20-layer network as a special case, the extra 30 layers could just learn to pass their input through unchanged (the identity function) and you'd match the shallow net exactly. So a deeper net should never be worse.

**The problem is optimisation, not capacity.** Gradient signal degrades as it propagates back through many layers. The network *could* represent the identity, but gradient descent can't find it. The layers don't know how to "do nothing".

ResNets fix this by making "do nothing" the *default*.

### The plain vs residual curve

<a id="the-plain-vs-residual-curve"></a>

```
Training error
    │
    │╲                    plain net (theory)
    │ ╲___________
    │      ╱              plain net (reality - gets worse!)
    │  ___╱
    │╲
    │ ╲______________     ResNet (keeps improving)
    └─────────────────── number of layers
```

---

## 5. The residual block

<a id="5-the-residual-block"></a>

### The maths, a plain block

<a id="the-maths-a-plain-block"></a>

In a normal ("plain") two-layer stack, the forward pass from activation $a^{[l]}$ is:

$$z^{[l+1]} = W^{[l+1]} a^{[l]} + b^{[l+1]}$$
$$a^{[l+1]} = g(z^{[l+1]})$$
$$z^{[l+2]} = W^{[l+2]} a^{[l+1]} + b^{[l+2]}$$
$$a^{[l+2]} = g(z^{[l+2]})$$

Information flows strictly along this "main path". Everything must pass through both weight matrices.

### The maths, a residual block

<a id="the-maths-a-residual-block"></a>

The residual block adds a **shortcut** (also called a **skip connection**): take $a^{[l]}$, and inject it directly into the block *just before the final ReLU*.

Only the last line changes:

$$\boxed{a^{[l+2]} = g\left(z^{[l+2]} + a^{[l]}\right)}$$

That's the entire idea. One addition.

```
          ┌──────────── shortcut / skip connection ────────────┐
          │                                                    ▼
 a[l] ──► Linear ──► ReLU ──► a[l+1] ──► Linear ──► z[l+2] ──► (+) ──► ReLU ──► a[l+2]
                    main path
```

### Why it works, the identity argument

<a id="why-it-works-the-identity-argument"></a>

This is the key derivation. Work through it slowly.

Suppose the two extra layers are useless for this task. What does the network need to do? Set their weights to zero:

$$W^{[l+2]} = 0, \qquad b^{[l+2]} = 0$$

Then $z^{[l+2]} = 0$, and:

$$a^{[l+2]} = g(z^{[l+2]} + a^{[l]}) = g(0 + a^{[l]}) = g(a^{[l]})$$

Now, $g$ is ReLU and $a^{[l]}$ is itself the output of a ReLU, so $a^{[l]} \geq 0$, which means $\text{ReLU}(a^{[l]}) = a^{[l]}$:

$$\boxed{a^{[l+2]} = a^{[l]}}$$

**The identity function.** The block learns to do nothing, perfectly, by driving weights toward zero, and weight decay already pushes weights toward zero. It's the *easiest* thing for the network to learn, not the hardest.

That's the whole insight your notes capture: **ResNets let a neural network learn the identity function very easily.** Adding layers can therefore never hurt. Worst case they become identity; best case they learn something useful.

### The gradient view (why gradients stop vanishing)

<a id="the-gradient-view-why-gradients-stop-vanishing"></a>

Write the block as $a^{[l+2]} = F(a^{[l]}) + a^{[l]}$ (ignoring the outer ReLU). Differentiate:

$$\frac{\partial a^{[l+2]}}{\partial a^{[l]}} = \frac{\partial F}{\partial a^{[l]}} + 1$$

That **`+1`** is everything. In backprop, gradients through a deep plain network are a product of many Jacobians, if each has magnitude < 1, the product shrinks exponentially and vanishes. With the `+1`, each term is *at least* 1, so gradient can flow all the way back to early layers along the shortcut path, unattenuated.

The skip connection is a **gradient highway**.

### The dimension problem, and where 1×1 convs first appear

<a id="the-dimension-problem-and-where-11-convs-first-appear"></a>

$z^{[l+2]} + a^{[l]}$ requires both tensors to be the same shape. Inside a block where nothing changes size, fine. But at every stage where you downsample or change channel count (e.g. 56×56×64 → 28×28×128), the shapes differ.

Solution, insert a projection matrix $W_s$ in the shortcut:

$$a^{[l+2]} = g\left(z^{[l+2]} + W_s\, a^{[l]}\right)$$

In practice $W_s$ is a **1×1 convolution with stride 2**: it changes the channel count and halves the spatial size in one cheap operation. (This is your first glimpse of why 1×1 convs matter; the next section covers them properly.)

Blocks are therefore of two types:
- **Identity block**: input and output shapes match, shortcut is a plain wire
- **Convolutional block**: shapes differ, shortcut has a 1×1 conv

### ResNet-34 and the bottleneck (ResNet-50+)

<a id="resnet-34-and-the-bottleneck-resnet-50"></a>

ResNet-34 is basically VGG's 3×3 design with skip connections added every two layers. But for deeper nets (50, 101, 152), a plain two-3×3 block gets expensive. So they use a **bottleneck block**: three layers:

$$1{\times}1 \ (\text{reduce channels}) \;\rightarrow\; 3{\times}3 \ (\text{process}) \;\rightarrow\; 1{\times}1 \ (\text{restore channels})$$

Example with 256 input channels:
- 1×1, 64 filters → squeeze 256 down to 64
- 3×3, 64 filters → the expensive spatial op, now on only 64 channels
- 1×1, 256 filters → expand back to 256 so the shortcut addition works

The expensive 3×3 runs on 64 channels instead of 256, roughly a **16× reduction** in that layer's cost. This is the same trick Inception uses, which you'll see shortly.

### Implementation

<a id="implementation-3"></a>

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)

        # shortcut: identity if shapes match, else 1x1 conv projection (W_s)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        identity = self.shortcut(x)          # a[l]  (possibly projected)
        out = self.relu(self.bn1(self.conv1(x)))   # a[l+1]
        out = self.bn2(self.conv2(out))            # z[l+2]
        out = out + identity                       # THE SKIP CONNECTION
        return self.relu(out)                      # a[l+2] = g(z[l+2] + a[l])
```

Note the addition happens **before** the final ReLU. That ordering is what makes the identity derivation work, get it wrong and you break the property.

**Exercise:** build a small ResNet for CIFAR-10 by stacking these blocks, then build the same depth without skip connections. Train both. Watch the plain one plateau at higher training loss. Seeing this yourself is worth ten readings of the paper.

### In industry

<a id="in-industry-3"></a>

ResNet is arguably **the most deployed neural architecture in history**. Skip connections are now in essentially everything, including Transformers.

- **Default backbone for detection and segmentation.** Faster R-CNN, Mask R-CNN, and most segmentation models use ResNet-50 or ResNet-101 as the feature extractor. If a company runs object detection, warehouse inventory, retail shelf auditing, defect inspection, there's a ResNet in it.
- **Medical imaging.** ResNet-50 fine-tuned on chest X-rays, retinal scans, histopathology slides is the standard baseline for FDA-cleared diagnostic aids. Depth matters here because the discriminating features are subtle.
- **Content moderation at scale.** Platforms classifying billions of uploaded images run ResNet-family models (usually distilled or quantised) because the accuracy-per-FLOP is well understood and the latency is predictable.
- **Skip connections beyond vision.** Every Transformer block, so every LLM you use, including this one, has residual connections around its attention and feed-forward sub-layers, for exactly the reason derived above. The idea generalised completely.

**Practical rule:** ResNet-50 pretrained on ImageNet is the sensible default starting point for almost any image classification problem. Start there, measure, then optimise.

---

# PART 3, 1×1 CONVOLUTIONS (Network in Network)

<a id="part-3-11-convolutions-network-in-network"></a>

---

## 6. What does a 1×1 convolution do?

<a id="6-what-does-a-11-convolution-do"></a>

### Plain English

<a id="plain-english-4"></a>

The first reaction is that a 1×1 filter is pointless. On a single-channel image, convolving with a 1×1 filter just multiplies every pixel by one number. Useless.

But images in a CNN aren't single-channel. A 1×1 filter applied to a **28×28×192** volume is really a **1×1×192** filter, it looks at one spatial position but **all 192 channels at once**, multiplies elementwise, sums, and applies ReLU.

So it's a fully-connected layer applied independently at every pixel location, operating across depth. That's why the paper is called **"Network in Network"**: you have a tiny neural network running at each spatial position.

```
28×28×192  ──[ 32 filters of 1×1×192 ]──►  28×28×32
```

Spatial size unchanged. **Channel count changed at will.**

### The maths

<a id="the-maths-3"></a>

For one output channel $k$ at position $(i,j)$:

$$z^{[k]}_{i,j} = \sum_{c=1}^{192} w^{[k]}_c \cdot a_{i,j,c} + b^{[k]}, \qquad \text{output} = g(z^{[k]}_{i,j})$$

Parameters for a 1×1 conv from $C_{in}$ to $C_{out}$ channels:

$$(1 \times 1 \times C_{in} + 1) \times C_{out}$$

For 192 → 32: $(192 + 1) \times 32 = 6{,}176$ parameters. Compare a 5×5 conv doing the same channel change: $(5 \times 5 \times 192 + 1) \times 32 = 153{,}632$. **25× cheaper.**

### The intuition, three distinct uses

<a id="the-intuition-three-distinct-uses"></a>

**1. Shrink channels (the "bottleneck").**
Pooling shrinks $n_H$ and $n_W$. Nothing shrinks $n_c$, until now. `28×28×192 → 28×28×32` cuts your depth by 6× at almost no cost. This is the single most important use and it powers both Inception and ResNet-50.

**2. Grow or keep channels, adding non-linearity.**
`28×28×192 → 28×28×192` with 192 filters keeps the shape but adds a full ReLU non-linearity and a learned cross-channel mixing. You've made the network more expressive without touching spatial resolution.

**3. Learned channel mixing.**
Think of your 192 channels as 192 different feature detectors. A 1×1 conv learns *combinations* of them, "fire strongly when edge-detector-7 and texture-detector-42 both fire". It's feature engineering across depth, learned rather than designed.

### The mental model that makes it click

<a id="the-mental-model-that-makes-it-click"></a>

A standard 3×3 conv does two jobs at once: it mixes **spatially** (across a 3×3 neighbourhood) and **across channels** (all input channels). A 1×1 conv does only the second job. Once you see convolution as two separable concerns, spatial mixing and channel mixing, the 1×1 conv stops being weird and becomes an obvious tool.

Push that idea further and you get **depthwise separable convolutions**: do all spatial mixing with one cheap depthwise 3×3, then all channel mixing with a 1×1. That's the core of MobileNet and EfficientNet.

### Implementation

<a id="implementation-4"></a>

```python
# A bottleneck: expensive path vs cheap path
import torch, torch.nn as nn

x = torch.randn(1, 192, 28, 28)

expensive = nn.Conv2d(192, 32, kernel_size=5, padding=2)
cheap     = nn.Sequential(
    nn.Conv2d(192, 16, kernel_size=1), nn.ReLU(),   # bottleneck: 192 -> 16
    nn.Conv2d(16,  32, kernel_size=5, padding=2),   # then the 5x5
)

def params(m): return sum(p.numel() for p in m.parameters())
print("5x5 direct:", params(expensive))   # ~153k
print("1x1 + 5x5 :", params(cheap))       # ~16k
print(expensive(x).shape, cheap(x).shape) # both (1, 32, 28, 28)
```

### In industry

<a id="in-industry-4"></a>

1×1 convs are load-bearing infrastructure in nearly every efficient model shipped today:

- **Mobile and edge deployment.** MobileNet, EfficientNet, and ShuffleNet are built almost entirely out of 1×1 convs plus depthwise 3×3s. Every phone camera feature, portrait mode segmentation, scene classification, real-time filters, runs on this. When you deploy a model to a phone or a Jetson board, you are deploying a stack of 1×1 convolutions.
- **Detection heads.** YOLO's prediction head is 1×1 convs mapping a feature map to per-anchor box coordinates and class scores. Feature Pyramid Networks use 1×1 convs to align channel counts across pyramid levels before merging.
- **Model compression for cost control.** A very common production optimisation: insert 1×1 bottlenecks into an over-parameterised model to cut inference cost 5–10× with a 1–2% accuracy loss. When you're serving millions of images a day, that's a direct and large infrastructure bill reduction.
- **Channel attention (Squeeze-and-Excitation).** SE blocks, in EfficientNet and most modern nets, use 1×1 convs to compute per-channel importance weights. "Which of my 192 feature detectors matter for *this* image?"

---

# PART 4, INCEPTION NETWORKS

<a id="part-4-inception-networks"></a>

---

## 7. Inception motivation

<a id="7-inception-motivation"></a>

### Plain English

<a id="plain-english-5"></a>

Designing a CNN forces constant decisions: 1×1 or 3×3 or 5×5? Conv or pool? Every layer, another guess.

Inception's answer: **stop choosing. Do all of them, and let the network learn which to use.**

An Inception module takes an input volume and runs four operations *in parallel*, then **concatenates** all the outputs along the channel axis.

### The module (from your slide)

<a id="the-module-from-your-slide"></a>

Input: **28×28×192**

| Branch | Operation | Output |
|---|---|---|
| 1 | 1×1 conv, 64 filters | 28×28×**64** |
| 2 | 3×3 conv, same, 128 filters | 28×28×**128** |
| 3 | 5×5 conv, same, 32 filters | 28×28×**32** |
| 4 | max-pool, same, then 1×1 → 32 | 28×28×**32** |

Concatenate along channels:

$$64 + 128 + 32 + 32 = 256 \quad \Rightarrow \quad \textbf{28×28×256}$$

**Critical detail:** all branches use `same` padding so every output is 28×28. If the spatial sizes didn't match you couldn't concatenate. Even the max-pool branch uses stride 1 with padding, unusual, but necessary here.

### The intuition

<a id="the-intuition-1"></a>

You've replaced a design decision with a learnable one. During training, gradient descent will grow the weights of whichever branch is useful for this dataset and shrink the others. If 5×5 patterns matter, that branch dominates. If not, it fades. **You've made the architecture itself partly learned.**

The cost: it's expensive. Which is the next problem.

---

## 8. The computational cost problem, and the bottleneck fix

<a id="8-the-computational-cost-problem-and-the-bottleneck-fix"></a>

This is the most important calculation in the lecture. Do it by hand.

### The expensive version

<a id="the-expensive-version"></a>

Take just the 5×5 branch: **28×28×192 → 5×5 same, 32 filters → 28×28×32**

Count multiplications. Each of the $28 \times 28 \times 32$ output values requires a dot product over one filter's worth of input: $5 \times 5 \times 192$ multiplications.

$$\text{cost} = \underbrace{28 \times 28 \times 32}_{\text{output values}} \times \underbrace{5 \times 5 \times 192}_{\text{multiplies each}} = 25{,}088 \times 4{,}800 \approx \mathbf{120\ \text{million}}$$

120 million multiplications, for **one branch of one layer**. Stack dozens of these and it's unaffordable.

### The bottleneck version

<a id="the-bottleneck-version"></a>

Insert a 1×1 conv to shrink channels first:

$$28{\times}28{\times}192 \xrightarrow{\ 1\times1,\ 16\ \text{filters}\ } 28{\times}28{\times}16 \xrightarrow{\ 5\times5,\ 32\ \text{filters}\ } 28{\times}28{\times}32$$

**Step 1 cost:**
$$(28 \times 28 \times 16) \times (1 \times 1 \times 192) = 12{,}544 \times 192 \approx \mathbf{2.4\ \text{million}}$$

**Step 2 cost:**
$$(28 \times 28 \times 32) \times (5 \times 5 \times 16) = 25{,}088 \times 400 \approx \mathbf{10.0\ \text{million}}$$

**Total: ≈ 12.4 million** versus 120 million.

$$\boxed{\text{roughly } \frac{1}{10} \text{ the computational cost, same input and output shape}}$$

That middle 28×28×16 layer is the **bottleneck layer**: the narrow waist the data is squeezed through.

### Does shrinking to 16 channels destroy information?

<a id="does-shrinking-to-16-channels-destroy-information"></a>

The reasonable worry, and the empirical answer is no, *within reason*. The 192 channels are highly redundant; a learned linear projection to 16 dimensions retains most of what matters, much like PCA. Shrink too aggressively (192 → 2) and you will hurt performance. The bottleneck width is a hyperparameter, and the values in GoogLeNet (16, 32, 96) were tuned empirically.

**This is the general principle behind bottlenecks everywhere:** ResNet-50's blocks, MobileNet's inverted residuals, and autoencoder latent spaces all rest on the same bet, that high-dimensional feature volumes are compressible without meaningful loss.

### The full Inception module with bottlenecks

<a id="the-full-inception-module-with-bottlenecks"></a>

```
                    ┌──► 1×1 conv (64) ─────────────────────────┐
                    │                                            │
  28×28×192 ────────┼──► 1×1 (96) ──► 3×3 same (128) ───────────┼──► concat ──► 28×28×256
                    │                                            │
                    ├──► 1×1 (16) ──► 5×5 same (32) ────────────┤
                    │                                            │
                    └──► 3×3 maxpool same ──► 1×1 (32) ─────────┘
```

Two things to note:
- The 1×1 branch needs no bottleneck, it *is* one.
- In the pool branch, the 1×1 comes **after** pooling (pooling can't change channel count, so you'd otherwise concatenate all 192 input channels and dominate the output).

---

## 9. The Inception network (GoogLeNet)

<a id="9-the-inception-network-googlenet"></a>

### Plain English

<a id="plain-english-6"></a>

GoogLeNet is nine Inception modules stacked, with occasional max-pools to halve spatial dimensions, and a softmax at the end. Despite being deeper than VGG-16, it has only **~5 million parameters**: versus VGG's 138M, because it has no giant fully-connected layers. It replaces them with **global average pooling**: take a 7×7×1024 volume and average each channel down to a single number, giving a 1024-vector straight into the classifier.

### Auxiliary classifiers

<a id="auxiliary-classifiers"></a>

GoogLeNet has softmax outputs branching off *middle* layers, not just the end. During training the total loss is:

$$\mathcal{L}_{total} = \mathcal{L}_{main} + 0.3\,\mathcal{L}_{aux1} + 0.3\,\mathcal{L}_{aux2}$$

Purpose: inject gradient directly into the middle of the network so early layers get a strong training signal, and act as a regulariser. These branches are **discarded at inference**: they exist only for training.

(This was a pre-ResNet workaround for the vanishing gradient problem. Skip connections solved it more elegantly, and auxiliary heads largely disappeared afterward. Worth knowing so you recognise them in old code.)

### And the name

<a id="and-the-name"></a>

The paper's citation is *"We need to go deeper"*, the Inception meme. That's the joke on your slide. It's a genuine footnote in the original paper.

### Implementation

<a id="implementation-5"></a>

```python
import torch
import torch.nn as nn

def conv_bn(in_c, out_c, **kw):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, bias=False, **kw),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class InceptionModule(nn.Module):
    def __init__(self, in_c, n1, n3_red, n3, n5_red, n5, pool_proj):
        super().__init__()
        self.b1 = conv_bn(in_c, n1, kernel_size=1)

        self.b2 = nn.Sequential(
            conv_bn(in_c, n3_red, kernel_size=1),            # bottleneck
            conv_bn(n3_red, n3, kernel_size=3, padding=1),
        )
        self.b3 = nn.Sequential(
            conv_bn(in_c, n5_red, kernel_size=1),            # bottleneck
            conv_bn(n5_red, n5, kernel_size=5, padding=2),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),            # 'same' pooling
            conv_bn(in_c, pool_proj, kernel_size=1),         # after pooling
        )

    def forward(self, x):
        # concatenate along the CHANNEL dimension (dim=1 in PyTorch NCHW)
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

# The module from the slides: 28x28x192 -> 28x28x256
m = InceptionModule(192, n1=64, n3_red=96, n3=128, n5_red=16, n5=32, pool_proj=32)
print(m(torch.randn(1, 192, 28, 28)).shape)   # torch.Size([1, 256, 28, 28])
```

Run that and confirm you get 256 channels. Then change a filter count and predict the new output before running it. When you can do that reliably, you understand Inception.

### In industry

<a id="in-industry-5"></a>

- **Inception-v3 as a production workhorse.** Better accuracy-per-FLOP than VGG made it a standard choice for large-scale image classification services. Google Photos search, early versions of Cloud Vision API, and countless internal classifiers were built on it.
- **The FID score.** If you evaluate an image generation model, GAN or diffusion, you almost certainly use **Fréchet Inception Distance**, which measures the distance between real and generated image distributions in Inception-v3's feature space. Every generative model paper reports it. Inception-v3 is permanently embedded in the field's evaluation infrastructure.
- **Multi-scale processing as a general pattern.** The core idea, process at several receptive-field sizes in parallel and merge, reappears constantly: Atrous Spatial Pyramid Pooling in DeepLab (semantic segmentation for autonomous driving and satellite imagery), FPN in detectors, and multi-head attention in Transformers (parallel heads with different learned views, concatenated).
- **Medical imaging.** Inception-v3 fine-tuned on retinal fundus photographs was the basis of Google's diabetic-retinopathy screening system, one of the first deep learning diagnostic tools to reach clinical deployment.

---

# PART 5, PRACTICAL ADVICE FOR USING CONVNETS

<a id="part-5-practical-advice-for-using-convnets"></a>

This part is the least mathematical and the most immediately valuable. It's what separates people who read papers from people who ship models.

---

## 10. Use open source implementations

<a id="10-use-open-source-implementations"></a>

### The advice

<a id="the-advice"></a>

1. Use architectures published in the literature
2. Use open source implementations wherever possible
3. Use pretrained models and fine-tune on your dataset

### Why this is not laziness

<a id="why-this-is-not-laziness"></a>

Reproducing a published architecture from the paper alone is genuinely difficult. Papers omit things: exact learning rate schedules, weight initialisation, augmentation details, batch norm placement, warmup epochs. Getting ResNet-50 to its published ImageNet accuracy from scratch takes weeks of GPU time and a lot of undocumented tribal knowledge.

More to the point: **pretrained weights represent compute you cannot afford.** Training ResNet-50 on ImageNet costs thousands of GPU-hours. Someone already paid that. Downloading the result costs you 100MB.

### How to actually do it

<a id="how-to-actually-do-it"></a>

```python
# Option 1: torchvision (classic architectures)
import torchvision.models as models
resnet = models.resnet50(weights='IMAGENET1K_V2')
effnet = models.efficientnet_b0(weights='IMAGENET1K_V1')

# Option 2: timm - the best library for vision backbones, ~1000 models
import timm
model = timm.create_model('resnet50', pretrained=True, num_classes=5)
print(timm.list_models('*efficientnet*', pretrained=True))

# Option 3: HuggingFace
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
```

`timm` (PyTorch Image Models) is what practitioners actually use. Learn it.

### In industry

<a id="in-industry-6"></a>

The build-vs-adopt decision is a real budget question. A startup that trains its own backbone from scratch is usually wasting six months. The organisations that legitimately train from scratch, foundation model labs, a handful of autonomous driving teams, do so because they have data at a scale where ImageNet pretraining stops helping.

**Everyone else fine-tunes.** Including teams at very large companies.

---

## 11. Transfer learning

<a id="11-transfer-learning"></a>

### Plain English

<a id="plain-english-7"></a>

Someone trained a network on 1.2 million ImageNet images. Its early layers learned edges, corners, textures, colours. Its middle layers learned shapes and parts. Those features are **not specific to ImageNet's 1000 classes**: edges are edges whether you're looking at cats or chest X-rays or cracked concrete.

So: keep those learned features, throw away the final classification layer, bolt on your own, and train.

### The three regimes (your slide shows all three)

<a id="the-three-regimes-your-slide-shows-all-three"></a>

**Regime 1, Very little data (tens to a few hundred images)**

Freeze *everything* except the final layer. Replace the 1000-way softmax with your own (say, 3 classes) and train only that.

$$\text{Train: } W^{[L]}, b^{[L]} \qquad \text{Freeze: all earlier layers}$$

Powerful trick here: since the frozen part never changes, **precompute its output once** and save to disk. Then you're training a tiny logistic regression on fixed feature vectors. Seconds per epoch, runs on a CPU.

**Regime 2, Moderate data (thousands of images)**

Freeze the early layers, unfreeze the last few conv blocks plus the head. Early layers hold generic features you want to keep; later layers hold ImageNet-specific features worth adapting.

**Regime 3, Lots of data (tens of thousands+)**

Unfreeze everything. Use the pretrained weights as **initialisation** rather than a fixed feature extractor, and fine-tune the whole network with a small learning rate.

### The rule of thumb

<a id="the-rule-of-thumb"></a>

$$\text{more data} \;\Longrightarrow\; \text{fewer frozen layers}$$

### Implementation

<a id="implementation-6"></a>

```python
import torch, torch.nn as nn, torchvision.models as models

model = models.resnet50(weights='IMAGENET1K_V2')

# --- Regime 1: freeze everything, retrain the head only ---
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 3)   # new head: 3 classes
# newly created layers default to requires_grad=True

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# --- Regime 2: unfreeze the last block too, with discriminative LRs ---
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # small LR: don't wreck features
    {'params': model.fc.parameters(),     'lr': 1e-3},  # larger LR: head is random
])
```

**Two mistakes to avoid:**

1. **Learning rate too high when fine-tuning.** A large LR destroys pretrained features in the first few batches ("catastrophic forgetting"). Use 10–100× smaller than you would training from scratch.
2. **Wrong input normalisation.** Pretrained models expect ImageNet statistics. Use them or your features will be subtly wrong:
   ```python
   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
   ```

### In industry

<a id="in-industry-7"></a>

Transfer learning is **the default workflow**, not an optimisation. Realistic examples:

- **Agricultural disease detection.** A team has 800 photos of diseased crop leaves. Training from scratch is hopeless. Fine-tuning EfficientNet-B0 gets 94% accuracy in an afternoon on one GPU. This is a very common Indian agri-tech pattern.
- **Manufacturing QC.** A factory needs to spot a specific defect. They have 200 defective examples because defects are rare. Freeze a ResNet backbone, train a head, deploy. Weeks, not years.
- **Retail / e-commerce catalogue tagging.** Auto-tagging product images by category, colour, and style. Fine-tuned backbones handle the long tail of categories where you'll never have millions of examples per class.
- **Medical imaging.** Nearly universal, because labelled medical data is scarce and expensive (a radiologist's time). ImageNet pretraining transfers surprisingly well to X-rays and pathology despite the domain gap.

**The real business insight:** transfer learning collapses the data requirement for a working vision model from millions of images to hundreds. That's what makes computer vision economically viable for small companies at all.

---

## 12. Data augmentation

<a id="12-data-augmentation"></a>

### Plain English

<a id="plain-english-8"></a>

Computer vision is almost always data-starved. Augmentation manufactures more training data by applying transformations that change the pixels but not the label. A flipped cat is still a cat.

The network sees a slightly different version each epoch, so it can't memorise specific images, it has to learn features that survive the transformation. That's regularisation.

### Common methods (from your slides)

<a id="common-methods-from-your-slides"></a>

**Mirroring**: horizontal flip. Nearly free, almost always safe.

**Random cropping**: take a random sub-region and resize. Teaches translation and scale invariance. The single most effective augmentation for classification.

**Rotation, shearing, local warping**: geometric distortions. Use carefully: sensible for satellite or microscope images (no canonical orientation), risky for text or road signs.

**Colour shifting**: perturb the R, G, B channels:

$$R \mathrel{+}= \delta_R, \quad G \mathrel{+}= \delta_G, \quad B \mathrel{+}= \delta_B$$

with $\delta$ drawn from a small random distribution. Makes the model robust to different lighting, cameras, and white balance. **PCA colour augmentation** (from the AlexNet paper, sometimes "PCA jittering") is the principled version: perturb along the principal components of the RGB distribution in your dataset, so shifts follow natural colour variation.

### Critical: augmentations must preserve the label

<a id="critical-augmentations-must-preserve-the-label"></a>

This is where people break their models.

- Horizontal flip on a **cat photo**: fine.
- Horizontal flip on a **digit "2"** or the letter "R": **you have created a wrong label.**
- Vertical flip on a **street scene**: nonsense; upside-down cars aren't in your test set.
- Vertical flip on a **satellite image**: fine, no canonical "up".
- Heavy colour shift on **flower species classification**: harmful, because colour *is* the signal.

**Always ask: would a human still give this the same label? And does this variation actually occur in my test distribution?** Augmenting for variation that never happens in deployment wastes capacity.

### Implementation

<a id="implementation-7"></a>

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # random crop + resize
    transforms.RandomHorizontalFlip(p=0.5),               # mirroring
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.05),     # colour shifting
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validation/test: NO random augmentation. Deterministic only.
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

**Never augment your validation or test set randomly.** Your metrics become noisy and non-comparable across runs.

Augmentation runs on the CPU in dataloader workers while the GPU trains. If your GPU utilisation is low, your augmentation pipeline is probably the bottleneck, raise `num_workers`, or move augmentation to the GPU with NVIDIA DALI or Kornia.

### Modern augmentations worth knowing

<a id="modern-augmentations-worth-knowing"></a>

Beyond the lecture's list, these are standard in current practice:

- **Cutout / Random Erasing**: black out a random rectangle. Forces the model to use the whole object, not one discriminative patch.
- **Mixup**: blend two images and their labels: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$. Sounds absurd, works remarkably well.
- **CutMix**: paste a patch of one image into another, mixing labels proportionally to area.
- **RandAugment / AutoAugment**: learned or randomly sampled augmentation policies. `transforms.RandAugment()` is one line and usually beats hand-tuned pipelines.

### In industry

<a id="in-industry-8"></a>

- **Autonomous driving.** Augmentation simulates the conditions you can't collect enough of: night, rain, fog, glare, motion blur. Some teams go further and use rendered synthetic data for genuinely rare events (a child running into the road).
- **Medical imaging.** Elastic deformations are standard for tissue and organ images because real anatomy varies elastically. Rotation and flipping are safe for histopathology (slides have no canonical orientation) but not for chest X-rays (left/right matters, flipping creates anatomically wrong images with situs inversus).
- **Retail and OCR.** Perspective warping, synthetic shadows, and background compositing to handle photos taken at odd angles under bad shop lighting.
- **The economics.** Labelling is the dominant cost in most vision projects. Augmentation is a compute-for-labels trade, cheap GPU cycles substituting for expensive human annotation. On a small dataset, a good augmentation pipeline routinely adds 5–15 percentage points of accuracy for zero labelling cost. It is usually the highest-ROI thing you can do.

---

## 13. The state of computer vision

<a id="13-the-state-of-computer-vision"></a>

### Data vs hand engineering

<a id="data-vs-hand-engineering"></a>

The final conceptual idea in the lecture, and it's a good mental model for the whole field.

There are **two sources of knowledge** you can put into a machine learning system:

1. **Labelled data**
2. **Hand-engineered features, network architecture, and other components**

These substitute for each other. When you have lots of data, you can let the network learn everything and use simple architectures. When you have little data, you have to inject human knowledge, clever architectures, hand-designed features, heavy augmentation, strong priors.

Positioning a few tasks on that spectrum:

```
  little data ◄──────────────────────────────────────────► lots of data
  more hand-engineering                             less hand-engineering
  ("hacks")

  object detection ──── image recognition ──── speech recognition
```

**Object detection** sits at the data-poor end: bounding-box labels are expensive, so detection architectures (anchor boxes, NMS, FPN, multi-stage heads) are full of hand-designed machinery.

**Speech recognition** sits at the data-rich end: audio with transcripts is abundant, so modern speech systems are comparatively simple end-to-end models.

**Image recognition** is in between.

The historical arc supports this. As datasets grew, hand-engineered components kept getting deleted, SIFT and HOG features disappeared, then hand-designed architectures gave way to searched ones (NAS, EfficientNet), and anchor boxes are being dropped by DETR-style detectors. Data replaces engineering when data is available.

**Your practical takeaway:** know where your problem sits. With 500 images, you need transfer learning, aggressive augmentation, and a small architecture with strong inductive biases. With 5 million images, use a big generic model and get out of its way. Choosing the wrong strategy for your data regime is one of the most common and most expensive mistakes in applied ML.

### Tips for benchmarks and competitions

<a id="tips-for-benchmarks-and-competitions"></a>

Your slide lists these, and it's important to understand **why they're flagged as competition tricks specifically.**

**1. Ensembling.** Train several networks independently (different seeds, different architectures) and average their outputs:

$$\hat{y} = \frac{1}{N}\sum_{i=1}^{N} \hat{y}_i$$

Individual models make partly uncorrelated errors; averaging cancels them. Reliably worth 1–2% on a benchmark.

**2. Multi-crop at test time (Test-Time Augmentation).** Run the classifier on several transformed versions of each test image, the standard "10-crop" is four corners plus centre, each also mirrored, and average the predictions. More robust, since a single crop might miss the object.

### The honest caveat

<a id="the-honest-caveat"></a>

**These help you win Kaggle. They will get you into trouble in production.**

- Ensembling 5 models costs 5× the inference compute, 5× the memory, 5× the deployment complexity, and 5× the serving bill, for maybe 1.5% accuracy.
- 10-crop TTA costs **10× the latency per image**. If you're serving real-time requests, that's usually disqualifying.

In a competition, accuracy is the only metric and compute is free. In production you're optimising accuracy *subject to* latency, cost, and maintainability constraints. Almost nobody deploys a 5-model ensemble with 10-crop TTA.

Where they *are* used in production: **offline batch processing** where latency doesn't matter (nightly re-scoring of a catalogue), and **high-stakes low-volume decisions** where accuracy dominates cost (a medical screening system processing hundreds of cases a day, not millions). Also **knowledge distillation**: train a big ensemble, then train a single small model to mimic its outputs, and deploy the small one. That's how you keep some of the accuracy gain without the serving cost.

---

# PART 6, YOUR IMPLEMENTATION ROADMAP

<a id="part-6-your-implementation-roadmap"></a>

Reading this once gives you vocabulary. Building things gives you understanding. In order:

### Week 1, Get the shapes in your fingers

<a id="week-1-get-the-shapes-in-your-fingers"></a>
1. Implement LeNet-5 from scratch in PyTorch. Train on MNIST to ~99%.
2. For every layer, predict the output shape before you run it. Verify with `print(x.shape)`.
3. Write a function that computes the parameter count of a conv layer, and check it against `sum(p.numel() for p in layer.parameters())`.

### Week 2, Blocks, not layers

<a id="week-2-blocks-not-layers"></a>
4. Implement `vgg_block()` and build VGG-11 for CIFAR-10.
5. Implement `ResidualBlock` and build a ResNet-18 for CIFAR-10.
6. **Run the key experiment:** train a 20-layer plain net and a 20-layer ResNet on CIFAR-10 and plot both *training* losses. Watch the plain net do worse. This is the single most instructive experiment in the whole lecture.

### Week 3, Efficiency

<a id="week-3-efficiency"></a>
7. Implement the bottleneck cost comparison from Part 4 and confirm the ~10× number yourself.
8. Implement `InceptionModule` and verify 28×28×192 → 28×28×256.
9. Implement a depthwise separable conv and compare its FLOPs to a standard conv.

### Week 4, The actual job

<a id="week-4-the-actual-job"></a>
10. Pick a real small dataset (200–2000 images), something you care about.
11. Fine-tune a pretrained ResNet-50. Try all three transfer-learning regimes and compare.
12. Add an augmentation pipeline. Measure the accuracy delta with and without it.
13. Measure inference latency. Then try to cut it by 3× and see what accuracy costs.

Step 13 is the one most people skip, and it's the one that makes you employable.

---

## The five ideas, if you remember nothing else

<a id="the-five-ideas-if-you-remember-nothing-else"></a>

1. **$n_H, n_W \downarrow$ and $n_c \uparrow$**: the universal shape of a CNN.
2. **$a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$**: skip connections make identity easy to learn, so depth stops hurting. This idea escaped vision entirely and is now in every Transformer.
3. **1×1 convolutions control channel depth cheaply**: the bottleneck, and the foundation of every efficient architecture.
4. **Transfer learning is the default**, not a shortcut. Fine-tune, don't train from scratch.
5. **Know your data regime.** Little data means more human-injected structure; lots of data means get out of the model's way.

---

## Papers, in reading order

<a id="papers-in-reading-order"></a>

| Paper | Year | The one idea |
|---|---|---|
| LeCun et al., Gradient-based learning applied to document recognition | 1998 | The CONV-POOL-FC template |
| Krizhevsky et al., ImageNet classification with deep CNNs | 2012 | Depth + ReLU + dropout + GPUs works |
| Simonyan & Zisserman, Very deep convolutional networks | 2015 | Uniform 3×3 blocks, repeated |
| Lin et al., Network in network | 2013 | The 1×1 convolution |
| Szegedy et al., Going deeper with convolutions | 2014 | Parallel multi-scale + bottlenecks |
| He et al., Deep residual learning for image recognition | 2015 | Skip connections |

Read them in that order and each one reads as a direct response to the previous one's limitation. That narrative is the actual history of the field.
