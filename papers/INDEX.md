# Papers

Summaries of papers I have read, written so that a year later I can recall what
each contributed without reopening the PDF.

A paper note is not an abstract. It records the problem, the idea, the numbers
worth remembering, what I would question, and what I would actually use.

**Template:** [PAPER_NOTE_TEMPLATE.md](../_templates/PAPER_NOTE_TEMPLATE.md)

---

## Index

- [Reading log](#reading-log)
- [By topic](#by-topic)
- [Reading list](#reading-list)
- [How to read a paper](#how-to-read-a-paper)

---

## Reading log

<a id="reading-log"></a>

| Year | Paper | One line contribution | Tags | Status |
| --- | --- | --- | --- | --- |
| | _No notes written yet_ | | | |

Sorted by publication year, not reading date. Within a subfield, papers make far
more sense read in the order they were written.

## By topic

<a id="by-topic"></a>

| Topic | Papers |
| --- | --- |
| Architectures | none yet |
| Optimization | none yet |
| Transformers | none yet |
| Generative | none yet |

## Reading list

<a id="reading-list"></a>

Queued, not yet read. Grouped so each cluster can be read as a conversation.

**Convolutional architectures**

| Year | Paper | Why |
| --- | --- | --- |
| 1998 | LeCun et al., Gradient-based learning applied to document recognition | The CONV, POOL, FC template |
| 2012 | Krizhevsky et al., ImageNet classification with deep CNNs | Depth plus ReLU plus dropout plus GPUs |
| 2013 | Lin et al., Network in network | The 1x1 convolution |
| 2014 | Szegedy et al., Going deeper with convolutions | Parallel multi-scale plus bottlenecks |
| 2015 | Simonyan and Zisserman, Very deep convolutional networks | Uniform 3x3 blocks, repeated |
| 2015 | He et al., Deep residual learning for image recognition | Skip connections |

**Sequence models and transformers**

| Year | Paper | Why |
| --- | --- | --- |
| 2014 | Sutskever et al., Sequence to sequence learning | The encoder decoder framing |
| 2015 | Bahdanau et al., Neural machine translation by jointly learning to align and translate | Attention |
| 2017 | Vaswani et al., Attention is all you need | The transformer |
| 2019 | Devlin et al., BERT | Bidirectional pretraining |

**Training and regularization**

| Year | Paper | Why |
| --- | --- | --- |
| 2014 | Srivastava et al., Dropout | The canonical regularizer |
| 2015 | Ioffe and Szegedy, Batch normalization | Made deep networks trainable |
| 2015 | Kingma and Ba, Adam | The default optimizer |

## How to read a paper

<a id="how-to-read-a-paper"></a>

Three passes, and most papers should stop after the first.

**Pass one, five minutes.** Title, abstract, figures, conclusion. Decide whether
to continue. Most papers do not survive this pass, and that is correct.

**Pass two, one hour.** Read for the idea, skipping proofs. Afterwards, write the
one line contribution from memory. If you cannot, you did not understand it.

**Pass three, several hours.** Only for papers you intend to implement or argue
with. Re-derive the key equations. Question every experimental choice. Note what
the paper does not say, especially in the appendix.

Do not read linearly. Abstract, then conclusion, then figures, then method, and
only then the related work section.
