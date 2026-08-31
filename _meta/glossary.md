# Glossary

Every term, one place. When a note uses a term for the first time it should
either define it inline or link here.

The most complete jargon reference currently lives inside
[Deep Learning Fundamentals, Appendix A](../01-foundations/neural-networks/deep-learning-fundamentals.md#appendix-a-complete-jargon-glossary).
That appendix will be migrated into this file as the repository grows, so that
terms are not tied to a single source document.

---

## Index

- [A to C](#a-to-c)
- [D to G](#d-to-g)
- [H to O](#h-to-o)
- [P to Z](#p-to-z)

---

## A to C

<a id="a-to-c"></a>

| Term | Meaning |
| --- | --- |
| **Activation** | The output of a neuron after its non-linear function. |
| **Activation function** | The non-linear squashing function: ReLU, sigmoid, tanh, GELU. |
| **Adam** | Adaptive Moment estimation. Momentum plus RMSprop plus bias correction. The default optimizer. |
| **Backpropagation** | Computing gradients by applying the chain rule backward through the network. |
| **Batch normalization** | Normalizing layer activations across the batch, which stabilises and speeds up training. |
| **Bias (parameter)** | The constant offset added inside a neuron. |
| **Bias (statistical)** | Underfitting. The model is too simple for the data. |
| **Bottleneck layer** | A layer that deliberately reduces channel or feature count to cut computation. |
| **Cost function** | The average loss across the whole training set. |

## D to G

<a id="d-to-g"></a>

| Term | Meaning |
| --- | --- |
| **Dropout** | Randomly zeroing activations during training to prevent co-adaptation. |
| **Epoch** | One full pass over the training set. |
| **Fine tuning** | Continuing training a pretrained model on your own data. |
| **Gradient descent** | Iteratively stepping parameters in the direction that reduces the cost. |

## H to O

<a id="h-to-o"></a>

| Term | Meaning |
| --- | --- |
| **Hyperparameter** | A setting chosen before training, not learned by gradient descent. |
| **Learning rate** | How large a step gradient descent takes. The hyperparameter that matters most. |
| **Loss function** | The error on a single training example. |
| **Overfitting** | High variance. Good on training data, bad on unseen data. |

## P to Z

<a id="p-to-z"></a>

| Term | Meaning |
| --- | --- |
| **Parameter** | A weight or bias, learned by gradient descent. |
| **Regularization** | Any technique that reduces overfitting. |
| **Residual connection** | A shortcut that adds a layer's input to its output, making identity easy to learn. |
| **Softmax** | Turns a vector of scores into probabilities that sum to one. |
| **Transfer learning** | Reusing features learned on one task for a different task. |
| **Variance** | Overfitting. The model is too sensitive to its training data. |
| **Vanishing gradient** | Gradients shrinking toward zero through depth, stalling learning in early layers. |
