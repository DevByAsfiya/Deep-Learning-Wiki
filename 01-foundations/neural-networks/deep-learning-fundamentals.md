# Deep Learning: Complete Beginner's Notes

> **Source:** personal course notes, converted from PDF
> **Level:** beginner to intermediate
> **Prerequisites:** basic Python, basic linear algebra (vectors and matrices), basic calculus (what a derivative is)
> **Covers:** neural network foundations, optimization and regularization, hyperparameter tuning, batch normalization, softmax, ML strategy, error analysis, transfer and multi-task learning, end-to-end deep learning
> **Related:** [Deep Learning for Computer Vision: CNN Part II](../../03-computer-vision/architectures/cnn-classic-architectures.md)

Detailed explanations of every topic in the syllabus, written in plain language,
with jargon decoded and a real project example for each concept.

Every section follows the same shape: **the idea in one line**, **jargon decoded**,
**deeper explanation** (intuition first, then the math), and a **real project example**.

Three running examples appear throughout so the ideas stay concrete:

| Example | What it is |
| --- | --- |
| **CatShop** | An e-commerce store classifying product photos and predicting which customers will buy. |
| **SaaSChurn** | A B2B SaaS product predicting which accounts will cancel next month. |
| **CreatorRank** | An influencer marketing platform predicting which creators will deliver good campaign ROI. |

You do not need to read this linearly, but Chapter 1 builds vocabulary that
Chapters 2 and 4 assume.

---

## Index

Every entry links to its section. Chapter headings are level 1, sections level 2,
and topics level 3.

- **[How to use these notes](#how-to-use-these-notes)**

### [Chapter 1: Neural Networks & Deep Learning](#chapter-1-neural-networks-deep-learning)

- **[1.1 Why Neural Networks?](#11-why-neural-networks)**
  - [1.1.1 What a neural network actually is](#111-what-a-neural-network-actually-is)
  - [1.1.2 The three main architectures: Standard NN, CNN, RNN](#112-the-three-main-architectures-standard-nn-cnn-rnn)
  - [1.1.3 Scale drives deep learning progress](#113-scale-drives-deep-learning-progress)
  - [1.1.4 Binary classification](#114-binary-classification)
  - [1.1.5 Notation](#115-notation)
  - [1.1.6 Logistic regression](#116-logistic-regression)
  - [1.1.7 The cost function](#117-the-cost-function)
  - [1.1.8 Gradient descent](#118-gradient-descent)
  - [1.1.9 Computation graphs and backpropagation](#119-computation-graphs-and-backpropagation)
  - [1.1.10 Gradient descent on m examples](#1110-gradient-descent-on-m-examples)
  - [1.1.11 Vectorization](#1111-vectorization)
  - [1.1.12 Vectorizing logistic regression](#1112-vectorizing-logistic-regression)
- **[1.2 Non-Linear over Linear Functions](#12-non-linear-over-linear-functions)**
  - [1.2.1 Neural network representation](#121-neural-network-representation)
  - [1.2.2 Vectorizing across multiple examples](#122-vectorizing-across-multiple-examples)
  - [1.2.3 Activation functions](#123-activation-functions)
  - [1.2.4 Sigmoid](#124-sigmoid)
  - [1.2.5 Tanh](#125-tanh)
  - [1.2.6 ReLU and Leaky ReLU](#126-relu-and-leaky-relu)
  - [1.2.7 Why non-linear activation functions at all?](#127-why-non-linear-activation-functions-at-all)
  - [1.2.8 Gradient descent for neural networks](#128-gradient-descent-for-neural-networks)
  - [1.2.9 Derivative formulas](#129-derivative-formulas)
  - [1.2.10 Zero initialization, and why it fails](#1210-zero-initialization-and-why-it-fails)
  - [1.2.11 Random initialization](#1211-random-initialization)
- **[1.3 Deep Neural Networks](#13-deep-neural-networks)**
  - [1.3.1 What is a deep neural network?](#131-what-is-a-deep-neural-network)
  - [1.3.2 Deep neural network notation](#132-deep-neural-network-notation)
  - [1.3.3 Forward propagation in a deep network](#133-forward-propagation-in-a-deep-network)
  - [1.3.4 Forward and backward functions](#134-forward-and-backward-functions)
  - [1.3.5 Forward propagation for layer l](#135-forward-propagation-for-layer-l)
  - [1.3.6 Backward propagation for layer l](#136-backward-propagation-for-layer-l)
  - [1.3.7 Parameters vs. hyperparameters](#137-parameters-vs-hyperparameters)
  - [1.3.8 Applied deep learning is an empirical process](#138-applied-deep-learning-is-an-empirical-process)

### [Chapter 2: Optimization Strategies in Neural Networks](#chapter-2-optimization-strategies-in-neural-networks)

- **[2.1 How to Improve Neural Networks](#21-how-to-improve-neural-networks)**
  - [2.1.1 Train / dev / test sets](#211-train-dev-test-sets)
  - [2.1.2 Mismatched train/test distribution](#212-mismatched-traintest-distribution)
  - [2.1.3 Bias and variance](#213-bias-and-variance)
  - [2.1.4 The basic recipe for machine learning](#214-the-basic-recipe-for-machine-learning)
  - [2.1.5 Regularization (logistic regression)](#215-regularization-logistic-regression)
  - [2.1.6 Regularization (neural networks)](#216-regularization-neural-networks)
  - [2.1.7 How does regularization prevent overfitting?](#217-how-does-regularization-prevent-overfitting)
  - [2.1.8 Dropout regularization](#218-dropout-regularization)
  - [2.1.9 Implementing dropout (inverted dropout)](#219-implementing-dropout-inverted-dropout)
  - [2.1.10 Making predictions at test time](#2110-making-predictions-at-test-time)
  - [2.1.11 Why does dropout work?](#2111-why-does-dropout-work)
  - [2.1.12 Data augmentation](#2112-data-augmentation)
  - [2.1.13 Early stopping](#2113-early-stopping)
  - [2.1.14 Normalizing training sets](#2114-normalizing-training-sets)
  - [2.1.15 Why normalize inputs?](#2115-why-normalize-inputs)
  - [2.1.16 Vanishing and exploding gradients](#2116-vanishing-and-exploding-gradients)
  - [2.1.17 Single neuron example (weight initialization intuition)](#2117-single-neuron-example-weight-initialization-intuition)
- **[2.2 Optimization Algorithms](#22-optimization-algorithms)**
  - [2.2.1 Batch vs. mini-batch gradient descent](#221-batch-vs-mini-batch-gradient-descent)
  - [2.2.2 Mini-batch gradient descent, the algorithm](#222-mini-batch-gradient-descent-the-algorithm)
  - [2.2.3 Training with mini-batch gradient descent, reading the cost curve](#223-training-with-mini-batch-gradient-descent-reading-the-cost-curve)
  - [2.2.4 Choosing your mini-batch size](#224-choosing-your-mini-batch-size)
  - [2.2.5 Exponentially weighted averages](#225-exponentially-weighted-averages)
  - [2.2.6 Understanding exponentially weighted averages](#226-understanding-exponentially-weighted-averages)
  - [2.2.7 Implementing exponentially weighted averages](#227-implementing-exponentially-weighted-averages)
  - [2.2.8 Bias correction](#228-bias-correction)
  - [2.2.9 Gradient descent with momentum](#229-gradient-descent-with-momentum)
  - [2.2.10 RMSprop](#2210-rmsprop)
  - [2.2.11 Adam optimization](#2211-adam-optimization)
  - [2.2.12 Learning rate decay](#2212-learning-rate-decay)
  - [2.2.13 The problem of local optima](#2213-the-problem-of-local-optima)
- **[2.3 Hyperparameter Tuning, Batch Normalization & Multi-Class Classification](#23-hyperparameter-tuning-batch-normalization-multi-class-classification)**
  - [2.3.1 The hyperparameter tuning process](#231-the-hyperparameter-tuning-process)
  - [2.3.2 Random values, not a grid](#232-random-values-not-a-grid)
  - [2.3.3 Using an appropriate scale to pick hyperparameters](#233-using-an-appropriate-scale-to-pick-hyperparameters)
  - [2.3.4 Hyperparameters for exponentially weighted averages](#234-hyperparameters-for-exponentially-weighted-averages)
  - [2.3.5 Re-test hyperparameters occasionally](#235-re-test-hyperparameters-occasionally)
  - [2.3.6 Batch normalization, normalizing activations in a network](#236-batch-normalization-normalizing-activations-in-a-network)
  - [2.3.7 Implementing batch norm](#237-implementing-batch-norm)
  - [2.3.8 Fitting batch norm into a neural network](#238-fitting-batch-norm-into-a-neural-network)
  - [2.3.9 Working with mini-batches](#239-working-with-mini-batches)
  - [2.3.10 Implementing gradient descent with batch norm](#2310-implementing-gradient-descent-with-batch-norm)
  - [2.3.11 Why does batch norm work?](#2311-why-does-batch-norm-work)
  - [2.3.12 Batch norm as regularization](#2312-batch-norm-as-regularization)
  - [2.3.13 Batch norm at test time](#2313-batch-norm-at-test-time)
  - [2.3.14 Multi-class classification and softmax regression](#2314-multi-class-classification-and-softmax-regression)
  - [2.3.15 The softmax layer and classifier](#2315-the-softmax-layer-and-classifier)
  - [2.3.16 The softmax loss function](#2316-the-softmax-loss-function)
- **[Deep Learning Frameworks](#deep-learning-frameworks)**
  - [Choosing a framework](#choosing-a-framework)
  - [The TensorFlow code example, explained](#the-tensorflow-code-example-explained)

### [Chapter 4: Structuring Machine Learning Projects](#chapter-4-structuring-machine-learning-projects)

- **[4.1 Introduction to ML Strategy](#41-introduction-to-ml-strategy)**
  - [4.1.1 Why ML strategy?](#411-why-ml-strategy)
  - [4.1.2 Orthogonalization](#412-orthogonalization)
- **[Setting Up Your Goal](#setting-up-your-goal)**
  - [4.1.3 Single number evaluation metric](#413-single-number-evaluation-metric)
  - [4.1.4 Satisficing and optimizing metrics](#414-satisficing-and-optimizing-metrics)
  - [4.1.5 Train / dev / test distributions](#415-train-dev-test-distributions)
  - [4.1.6 Size of dev and test sets](#416-size-of-dev-and-test-sets)
  - [4.1.7 When to change dev/test sets and metrics](#417-when-to-change-devtest-sets-and-metrics)
- **[Comparing to Human-Level Performance](#comparing-to-human-level-performance)**
  - [4.1.8 Why human-level performance?](#418-why-human-level-performance)
  - [4.1.9 Avoidable bias](#419-avoidable-bias)
  - [4.1.10 Understanding human-level performance](#4110-understanding-human-level-performance)
  - [4.1.11 Summary of bias/variance with human-level performance](#4111-summary-of-biasvariance-with-human-level-performance)
  - [4.1.12 Surpassing human-level performance](#4112-surpassing-human-level-performance)

### [Structuring ML Projects, Part II](#structuring-ml-projects-part-ii)

- **[1. Error Analysis](#1-error-analysis)**
  - [1.1 Carrying out error analysis](#11-carrying-out-error-analysis)
  - [1.2 Evaluating multiple ideas in parallel](#12-evaluating-multiple-ideas-in-parallel)
  - [1.3 Cleaning up incorrectly labelled data](#13-cleaning-up-incorrectly-labelled-data)
  - [1.4 Build your first system quickly, then iterate](#14-build-your-first-system-quickly-then-iterate)
- **[2. Mismatched Training and Dev/Test Sets](#2-mismatched-training-and-devtest-sets)**
  - [2.1 Training and testing on different distributions](#21-training-and-testing-on-different-distributions)
  - [2.2 Bias and variance with mismatched data distributions](#22-bias-and-variance-with-mismatched-data-distributions)
  - [2.3 The general formulation](#23-the-general-formulation)
  - [2.4 Addressing data mismatch](#24-addressing-data-mismatch)
  - [2.5 Artificial data synthesis](#25-artificial-data-synthesis)
- **[3. Learning from Multiple Tasks](#3-learning-from-multiple-tasks)**
  - [3.1 Transfer learning](#31-transfer-learning)
  - [3.2 Multi-task learning](#32-multi-task-learning)
- **[4. End-to-End Deep Learning](#4-end-to-end-deep-learning)**
  - [4.1 What is end-to-end deep learning?](#41-what-is-end-to-end-deep-learning)
  - [4.2 The face recognition (turnstile) example](#42-the-face-recognition-turnstile-example)
  - [4.3 More examples](#43-more-examples)
  - [4.4 Pros and cons of end-to-end deep learning](#44-pros-and-cons-of-end-to-end-deep-learning)
  - [4.5 Whether to use end-to-end deep learning](#45-whether-to-use-end-to-end-deep-learning)
- **[Quick recap of the four themes](#quick-recap-of-the-four-themes)**

### [Appendix A: Complete Jargon Glossary](#appendix-a-complete-jargon-glossary)

### [Appendix B: Decision Cheat Sheets](#appendix-b-decision-cheat-sheets)

- **[B.1 Which activation function?](#b1-which-activation-function)**
- **[B.2 My model isn't working, what do I check?](#b2-my-model-isnt-working-what-do-i-check)**
- **[B.3 First 60 minutes on any new problem](#b3-first-60-minutes-on-any-new-problem)**
- **[B.4 Sensible starting hyperparameters](#b4-sensible-starting-hyperparameters)**

### [Appendix C: A Suggested Practice Path](#appendix-c-a-suggested-practice-path)

---

*Detailed explanations of every topic in the syllabus, written in plain language, with jargon decoded and a real-project example for each concept.*

### How to use these notes

<a id="how-to-use-these-notes"></a>

Every section follows the same shape:

- **The idea in one line**: what the concept actually is.
- **Jargon decoded**: the intimidating words, unpacked.
- **Deeper explanation**: the intuition, then the math.
- **Real project example**: how you'd actually use this in something you build.

Three running examples appear throughout so the ideas stay concrete:

1. **CatShop**: an e-commerce store that needs to classify product photos and predict which customers will buy.
2. **SaaSChurn**: a B2B SaaS product predicting which accounts will cancel next month.
3. **CreatorRank**: an influencer marketing platform predicting which creators will deliver good campaign ROI.

You do not need to read this linearly. But Chapter 1 builds vocabulary that Chapters 2 and 4 assume.

## Chapter 1: Neural Networks & Deep Learning

<a id="chapter-1-neural-networks-deep-learning"></a>

### 1.1 Why Neural Networks?

<a id="11-why-neural-networks"></a>

#### 1.1.1 What a neural network actually is

<a id="111-what-a-neural-network-actually-is"></a>

**The idea in one line:** A neural network is a stack of very simple math functions that, when layered, can learn almost any mapping from input to output.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Neuron / unit** | One tiny function. It takes numbers in, multiplies each by a weight, adds them up, adds a bias, then squashes the result through a curve. That's it. |
| **Weight (w)** | How much a neuron cares about a particular input. Learned from data. |
| **Bias (b)** | A constant offset that lets the neuron shift its output up or down. Also learned. |
| **Layer** | A group of neurons that all look at the same inputs, in parallel. |
| **Deep** | More than one hidden layer. "Deep learning" just means "neural networks with several layers." |
| **Hidden layer** | Any layer between input and output. Called "hidden" because the training data never tells you what those values should be, the network invents them. |

**Deeper explanation:**

Start with the classic housing-price example. You want to predict price from house size.

The simplest model is a straight line: `price = w × size + b`. But prices don't go negative, so you clip the line at zero. That bent line, flat at zero, then rising, *is* a single neuron with a ReLU activation. One neuron, one input, one output.

Now add more inputs: size, number of bedrooms, zip code, neighbourhood wealth. You could wire every input straight to the price. But something more interesting happens if you insert a middle layer. The middle neurons might learn to represent things nobody told them about, "family size capacity" (from size + bedrooms), "walkability" (from zip code), "school quality" (from zip + wealth). Then the final neuron combines those *learned concepts* into a price.

Nobody hand-coded "walkability." The network discovered that combining certain inputs is useful, because doing so reduced its prediction error. **That's the whole magic of neural networks: they learn intermediate representations automatically.**

Given enough input-output pairs, neural networks are remarkably good at finding these mappings. The technical name for this property is the **universal approximation theorem**: a sufficiently large network can approximate essentially any continuous function.

**Real project example (CatShop):** You want to predict a product's expected weekly sales. Inputs: price, category, photo quality score, number of reviews, average rating, seller rating, days since listing. Instead of hand-engineering features like "value for money = rating / price," you feed the raw seven numbers into a network with two hidden layers of 16 units each. The hidden layers will construct their own equivalents of "value for money" and "social proof", usually better ones than you'd write by hand.

#### 1.1.2 The three main architectures: Standard NN, CNN, RNN

<a id="112-the-three-main-architectures-standard-nn-cnn-rnn"></a>

**The idea in one line:** The shape of your data determines the shape of your network.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Standard NN / Feedforward NN / MLP / Dense network** | Every neuron connects to every neuron in the next layer. Data flows one direction. Good for tabular data. |
| **CNN (Convolutional Neural Network)** | Uses small sliding filters that scan across an image. Good for anything with spatial structure. |
| **RNN (Recurrent Neural Network)** | Has a memory that carries forward through a sequence. Good for anything with time or order. |
| **Structured data** | Rows and columns. A database table. Each column has a clear meaning. |
| **Unstructured data** | Images, audio, raw text. Individual pixels/samples/characters have no standalone meaning. |

**Deeper explanation:**

- **Standard NN**: Use when your input is a fixed-length list of meaningful numbers. Real estate prices, ad click prediction, churn scores, credit risk. This is the workhorse for business data.
- **CNN**: Use for images. The key insight: a cat in the top-left corner is still a cat in the bottom-right. A CNN uses the same small filter everywhere on the image (this is called **parameter sharing** or **translation invariance**), so it needs far fewer parameters than a dense network and generalises far better. Also used for audio spectrograms and even some text tasks.
- **RNN**: Use when order matters and length varies. Speech (audio in, text out), machine translation, time series. Each step's output feeds back as input to the next step, giving the network a running memory. In practice you'll mostly see the improved variants, **LSTM** and **GRU**: which fix the plain RNN's tendency to forget things from long ago. (In 2020s practice, **Transformers** have largely displaced RNNs for language, but the sequence-modelling intuitions carry over directly.)
- **Hybrid/custom**: Autonomous driving takes camera images (CNN) plus radar (structured) plus temporal context (RNN) and fuses them.

**Real project example (CreatorRank):** One product, three architectures.

- Predicting campaign ROI from a creator's stats (follower count, engagement rate, past campaign results) → **Standard NN**, structured data.
- Detecting whether a creator's photos match a brand's aesthetic → **CNN**, unstructured images.
- Predicting whether a creator's engagement is trending up or down over 90 days of daily data → **RNN/temporal model**, sequence data.

#### 1.1.3 Scale drives deep learning progress

<a id="113-scale-drives-deep-learning-progress"></a>

**The idea in one line:** Deep learning took off because three things got big at the same time, data, compute, and better algorithms.

**Deeper explanation:**

Plot "amount of labelled data" on the x-axis and "performance" on the y-axis. You get a family of curves:

- **Traditional ML** (logistic regression, SVMs, random forests) rises quickly at first, then **plateaus**. Feeding it 10× more data barely helps.
- **Small neural network**: plateaus higher.
- **Medium neural network**: higher still.
- **Large neural network**: keeps climbing long after the others have flattened.

Two practical consequences:

1. **In the small-data regime, the ranking is unreliable.** With 1,000 examples, a well-tuned gradient-boosted tree may beat a neural network. Skill at feature engineering matters more than architecture. *Don't reach for deep learning by default on small tabular datasets.*
2. **In the big-data regime, network size is the lever.** To get better, you make the network bigger and feed it more data. This is why "scale" became the dominant story.

The third driver, **algorithms**: is often underrated. The switch from sigmoid to **ReLU** activations is the classic example. Sigmoid saturates: for large positive or negative inputs its slope is nearly zero, so learning crawls. ReLU has a constant slope of 1 for all positive inputs, so gradient descent runs much faster. That was a pure algorithmic change, no extra data or hardware, and it made training deep networks practical.

Fast computation also matters for a non-obvious reason: **it speeds up the idea → code → experiment → idea loop.** If a training run takes 10 minutes you'll try 30 ideas this week. If it takes 3 days you'll try two. Iteration speed is a first-class research asset.

**Real project example (SaaSChurn):** You have 4,000 customer accounts. That's small data. Start with gradient-boosted trees (XGBoost), it will likely beat a neural network and train in seconds. Revisit neural networks when you either (a) accumulate hundreds of thousands of accounts, or (b) want to fold in unstructured signals like support-ticket text, where deep learning has a genuine structural advantage.

#### 1.1.4 Binary classification

<a id="114-binary-classification"></a>

**The idea in one line:** Given an input, output a single probability between 0 and 1 answering a yes/no question.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Binary classification** | Two possible answers. Cat / not cat. Churn / no churn. |
| **Label (y)** | The correct answer, from your training data. 0 or 1. |
| **Prediction (ŷ, "y-hat")** | What the model outputs. A probability, e.g. 0.83. |
| **Feature vector (x)** | All the inputs for one example, stacked into a column of numbers. |
| **Training example** | One (x, y) pair. |

**Deeper explanation:**

Take a 64×64 colour image. A computer stores it as three grids of numbers, one for red, one for green, one for blue, each 64×64. To feed it into a standard neural network, you **unroll** (flatten) all those numbers into one long column vector. Length = 64 × 64 × 3 = **12,288**. So `n_x = 12288`, that's the dimension of the input feature vector.

The model outputs a single number ŷ = P(y=1 | x): "the probability that this image is a cat, given these pixels." To turn it into a decision, threshold it, usually at 0.5, but the threshold is a business choice, not a math one.

**Real project example (CatShop):** Fraudulent-listing detection. Input: 40 numbers describing a listing (price vs. category median, seller account age, number of images, text-similarity to known scams...). Output: probability the listing is fraudulent. You'd set the threshold *low*, maybe 0.15, because missing a fraud costs far more than a human spending 30 seconds reviewing a false alarm. **The model gives you a probability; you decide the threshold based on the relative cost of the two error types.**

#### 1.1.5 Notation

<a id="115-notation"></a>

**The idea in one line:** Consistent notation is the difference between reading a deep learning paper fluently and being lost.

**The standard conventions:**

| **Symbol** | **Meaning** |
| --- | --- |
| `m` | Number of training examples |
| `n_x` (or `n`) | Number of input features |
| `x⁽ⁱ⁾` | The i-th training example (a column vector of length n_x) |
| `y⁽ⁱ⁾` | The label for the i-th example |
| `X` | All inputs stacked as columns → shape **(n_x, m)** |
| `Y` | All labels in a row → shape **(1, m)** |
| `w` | Weight vector, shape (n_x, 1) |
| `b` | Bias, a single number |
| `z` | The pre-activation value: `z = wᵀx + b` |
| `a` | The activation: `a = σ(z)`. For the output layer, `a = ŷ` |
| `L(ŷ, y)` | **Loss**: error on ONE example |
| `J(w, b)` | **Cost**: average loss over ALL m examples |
| `[l]` superscript in square brackets | Layer number, e.g. `W[2]` |
| `(i)` superscript in round brackets | Example number, e.g. `x⁽³⁾` |

**Two things beginners trip on:**

1. **Loss vs. cost.** Loss is per-example. Cost is the average across the dataset. You minimise the cost. Many people use the words interchangeably in conversation; in code and in these notes, they're distinct.
2. **Column convention.** In this notation, `X` puts each *example* in a *column*, giving shape (n_x, m). Most libraries (pandas, scikit-learn, Keras) do the opposite, examples in *rows*, shape (m, n_x). Neither is wrong. **Just always check your shapes.** A huge fraction of deep learning bugs are shape bugs that don't crash, they just silently produce nonsense.

**Real project example:** Before every training run, print `X.shape` and `Y.shape` and assert they're what you expect. It takes ten seconds and saves entire afternoons.

#### 1.1.6 Logistic regression

<a id="116-logistic-regression"></a>

**The idea in one line:** The simplest possible classifier, a linear function squashed into the range (0, 1), and the atom that every neural network is built from.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Linear combination** | Multiply each input by a weight and add them up: `w₁x₁` `+ w₂x₂` `+... + b` |
| **Sigmoid (σ)** | The S-shaped squashing function: `σ(z) = 1 / (1 + e^(-z))` |
| **Parameters** | The things the model learns: w and b |

**Deeper explanation:**

Why can't you just use `ŷ = wᵀx + b` directly? Because that produces any real number, 4.7, −200, and probabilities must sit between 0 and 1. So you pass it through the sigmoid:

```text
z = wᵀx + b
ŷ = σ(z) = 1 / (1 + e^(-z))
```

Sigmoid's behaviour:

- `z` very large positive → `e^(-z)` ≈ 0 → σ(z) ≈ 1
- `z = 0` → σ(z) = 0.5
- `z` very large negative → `e^(-z)` huge → σ(z) ≈ 0

Smooth, monotonic, always between 0 and 1. Exactly what a probability needs.

**Real project example (SaaSChurn):** Logistic regression is genuinely the right first model. Twelve features (logins last 30 days, seats used vs. purchased, support tickets, days since last feature adoption,...), one weight each, plus a bias. It trains in milliseconds, and crucially **the weights are interpretable**: a large negative weight on "logins last 30 days" tells your customer success team exactly what to watch. Build this before you build anything deep. It is your baseline, and sometimes it's your final answer.

#### 1.1.7 The cost function

<a id="117-the-cost-function"></a>

**The idea in one line:** A single number measuring how wrong the model currently is, the thing you push downhill.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Loss function** | Error on one example |
| **Cost function** | Average loss over all examples |
| **Cross-entropy / log loss** | The standard loss for classification |
| **Convex** | Bowl-shaped. One global minimum, no local traps. |

**Deeper explanation:**

The obvious choice is squared error: `L = ½(ŷ − y)²`. It works for regression. For logistic regression it's a bad idea, combined with the sigmoid it produces a **non-convex** cost surface, full of local minima where gradient descent gets stuck.

Instead, use **binary cross-entropy**:

L(ŷ, y) = −[ y·log(ŷ) + (1−y)·log(1−ŷ) ]

This looks arbitrary until you plug in the two cases:

- **If y = 1:** the second term vanishes, leaving `L = −log(ŷ)`. To make loss small you need log(ŷ) large, so you need ŷ large, close to 1. Correct.
- **If y = 0:** the first term vanishes, leaving `L = −log(1−ŷ)`. To make loss small you need ŷ close to
0. Correct.

It also punishes confident mistakes brutally. Predicting 0.99 when the truth is 0 gives a loss of −log(0.01) ≈

4. 6. Predicting 0.5 gives 0.69. **Confidently wrong is much worse than uncertain.** This is a feature, it
forces the model to calibrate.

The cost is the average:

J(w, b) = (1/m) · Σᵢ L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)

And this cost *is* convex, so gradient descent will find the global minimum.

**Real project example (CatShop fraud):** Cross-entropy is the default and usually correct. But if fraud is 2% of your listings, the model can achieve 98% accuracy by predicting "not fraud" every time, and cross-entropy will happily let it, because 98% of examples are easy. The fix is **class weighting**: multiply the loss for fraud examples by, say, 20, so the model feels real pain for missing them. In Keras that's the `class_weight` argument. This is your first taste of the Chapter 4 lesson: *if the metric doesn't match what you actually care about, change the metric.*

#### 1.1.8 Gradient descent

<a id="118-gradient-descent"></a>

**The idea in one line:** Repeatedly take a small step in the direction that most reduces the cost.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Derivative / gradient** | The slope. "If I nudge this parameter up a tiny bit, how much does the cost change?" |
| **Learning rate (α, alpha)** | Step size. How far you move each iteration. |
| **Iteration / epoch** | One update step / one full pass through the training data. |
| **Converge** | Stop improving meaningfully, you've reached the bottom. |
| **∂J/∂w (written** `dw` **in code)** | The gradient of the cost with respect to w. |

**Deeper explanation:**

Picture the cost function as a landscape. The parameters (w, b) are your coordinates; the height is the cost. You're blindfolded and want to reach the lowest point. You feel the ground to find which direction slopes downhill most steeply, and take a step. Repeat.

```text
repeat until converged:
    w := w − α · (∂J/∂w)
    b := b − α · (∂J/∂b)
```

The minus sign matters. The gradient points **uphill**, so you move in the *opposite* direction.

The learning rate is the single most important hyperparameter you will tune:

- **Too small** → training takes forever. Thousands of iterations to move anywhere.
- **Too large** → you overshoot the minimum, bounce to the other side, overshoot again. The cost oscillates or explodes to NaN.
- **Just right** → smooth, steady decrease.

**How to diagnose it in practice:** plot cost against iteration number.

- Cost decreasing smoothly → good.
- Cost decreasing but painfully slowly → increase α.
- Cost jumping up and down → decrease α.
- Cost becomes NaN → α is far too large; drop it by 10×.

A good search strategy is to try α on a logarithmic ladder: 0.0001, 0.001, 0.01, 0.1, 1.0. Find the largest one that's still stable, then back off slightly.

**Real project example:** Whatever you're training, log the cost every N steps and look at the curve before you look at anything else. Ninety percent of "my model doesn't work" problems are visible in that one plot within the first 200 iterations.

#### 1.1.9 Computation graphs and backpropagation

<a id="119-computation-graphs-and-backpropagation"></a>

**The idea in one line:** Break a complicated calculation into simple steps, compute forward to get the answer, then walk backward applying the chain rule to get every derivative.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Computation graph** | A diagram of your calculation as a chain of simple operations. |
| **Forward propagation** | Left to right. Inputs → prediction → cost. |
| **Backward propagation (backprop)** | Right to left. Cost → gradients for every parameter. |
| **Chain rule** | If a affects b and b affects c, then a's effect on c = (a's effect on b) × (b's effect on c). Multiply the slopes along the path. |

**Deeper explanation:**

Take a toy function `J = 3(a + bc)`. Break it up:

```text
u = b · c
v = a + u
J = 3v
```

Forward pass with a=5, b=3, c=2: u = 6, v = 11, J = 33.

Now go backward. We want to know how much J changes if we nudge each input.

- `dJ/dv = 3`, bump v by 0.001, J goes up 0.003.
- `dJ/du = dJ/dv × dv/du = 3 × 1 = 3`
- `dJ/da = dJ/dv × dv/da = 3 × 1 = 3`
- `dJ/db = dJ/du × du/db = 3 × c = 6`
- `dJ/dc = dJ/du × du/dc = 3 × b = 9`

That's backpropagation. Nothing more mysterious than the chain rule, applied systematically right-to-left, reusing intermediate results so you don't recompute anything.

**Notation convention:** in code, the variable holding `∂J/∂w` is just called `dw`. The `∂J/∂` part is implied, you're always differentiating the final cost.

**Logistic regression's computation graph:**

Forward:

```text
z = w₁x₁ + w₂x₂ + b
a = σ(z)
L = −[y·log(a) + (1−y)·log(1−a)]
```

Backward:

```text
da = ∂L/∂a = −y/a + (1−y)/(1−a)
dz = ∂L/∂z = a − y          ← beautifully simple
```

```text
dw₁ = x₁ · dz
dw₂ = x₂ · dz
db  = dz
```

That `dz = a − y` is worth memorising. All the algebra of the sigmoid derivative and the log terms cancels, leaving **prediction minus truth**. Big error → big gradient → big update. Small error → small update. It's the same simple form in linear regression, logistic regression, and softmax classification, which is not a coincidence, it falls out of the mathematics of matching the right loss to the right output activation.

**Real project example:** You will almost never write backprop by hand. PyTorch and TensorFlow do it automatically (this feature is called **autograd** / **automatic differentiation**). But understanding it is what lets you diagnose vanishing gradients, choose activation functions intelligently, and read error messages that mention "the gradient." Write it by hand once, in numpy, for a two-layer network. Then never again.

#### 1.1.10 Gradient descent on m examples

<a id="1110-gradient-descent-on-m-examples"></a>

**The idea in one line:** Compute the gradient for every example, average them, take one step.

**Deeper explanation:**

The cost is the average of individual losses, and derivatives are linear, so the gradient of the cost is the average of individual gradients:

∂J/∂w = (1/m) · Σᵢ ∂L(a⁽ⁱ⁾, y⁽ⁱ⁾)/∂w

Written naively with loops, one iteration looks like this:

```text
J = 0; dw1 = 0; dw2 = 0; db = 0
for i in range(m):                    # loop over examples
    z = w1*x1[i] + w2*x2[i] + b
    a = sigmoid(z)
    J += -(y[i]*log(a) + (1-y[i])*log(1-a))
    dz = a - y[i]
    dw1 += x1[i] * dz                 # loop over features (implicit)
    dw2 += x2[i] * dz
    db  += dz
J /= m; dw1 /= m; dw2 /= m; db /= m
w1 -= alpha * dw1
w2 -= alpha * dw2
b  -= alpha * db
```

**This code is correct and unusably slow.** Two nested loops, one over m examples, one over n features. With m = 1,000,000 and n = 12,288 that's twelve billion Python-level operations per single gradient step. In Python, that is hours per step.

Which brings us to the most important engineering idea in the chapter.

#### 1.1.11 Vectorization

<a id="1111-vectorization"></a>

**The idea in one line:** Replace explicit loops with whole-array operations, and get a 100–300× speedup for free.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Vectorization** | Expressing a computation as operations on whole arrays instead of element-by-element loops. |
| **SIMD** | *Single Instruction, Multiple Data*, CPU/GPU hardware that applies one operation to many numbers simultaneously. |
| **Broadcastin g** | NumPy automatically stretching a smaller array to match a bigger one's shape. |

**Deeper explanation:**

Compare:

```python
# Non-vectorized
z = 0
for i in range(n):
    z += w[i] * x[i]
z += b
# Vectorized
z = np.dot(w, x) + b
```

Same result. On a million-element vector, the second is roughly 300× faster. Why? The Python loop runs one multiply at a time through the interpreter. `np.dot` drops into optimised C/Fortran (BLAS) that uses SIMD instructions to process 8 or 16 numbers per clock cycle, keeps data in cache, and can use multiple cores. GPUs take this further with thousands of parallel cores.

**The rule:** whenever you're about to write an explicit `for` loop over data in deep learning code, stop and ask whether numpy can do it as an array operation. Usually it can.

**Broadcasting** is the companion trick. If you add a (3,1) column vector to a (3,4) matrix, numpy silently copies the column across all four columns and adds elementwise. This is how you add a bias vector to a whole batch at once. It's enormously convenient and also a common source of silent bugs, a (5) array and a (5,1) array behave differently. **Always use explicit shapes:** `np.random.randn(5, 1)`**, not**

`np.random.randn(5)`**.** And use `assert(a.shape == (5,1))` liberally.

**Real project example:** A team was computing pairwise similarity between 50,000 creators using a Python double loop, 2.5 billion iterations, estimated 40 hours. Rewritten as a single matrix multiplication of the normalised embedding matrix with its own transpose, it ran in 90 seconds. Same math, same result. This is not an exotic optimisation; it's the baseline expectation for numerical code.

#### 1.1.12 Vectorizing logistic regression

<a id="1112-vectorizing-logistic-regression"></a>

**The idea in one line:** With the right matrix layout, one full gradient descent step over a million examples is about five lines of numpy with no loops at all.

**Deeper explanation:**

Stack all examples as columns: `X` has shape (n_x, m). Then:

**Forward pass, all m examples at once:**

```python
Z = np.dot(w.T, X) + b      # (1, m).  b broadcasts across all m columns
A = sigmoid(Z)              # (1, m).  every prediction
```

**Backward pass, all m gradients at once:**

```python
dZ = A - Y                          # (1, m)
dw = (1/m) * np.dot(X, dZ.T)        # (n_x, 1)
db = (1/m) * np.sum(dZ)             # scalar
```

**Update:**

```text
w = w - alpha * dw
b = b - alpha * db
```

That's the entire algorithm. Both loops, over examples and over features, have disappeared into two matrix multiplications. The only loop that remains is over gradient descent iterations, and that one is genuinely sequential, so it has to stay.

Take a moment to verify the shapes:

- `w.T` is (1, n_x), `X` is (n_x, m) → product is (1, m). ✓
- `X` is (n_x, m), `dZ.T` is (m, 1) → product is (n_x, 1), matching w. ✓

**Real project example:** This vectorized form is the template for everything that follows. Deep networks are the same five lines repeated per layer with the shapes changed. If you internalise the shape bookkeeping here, deep networks become straightforward rather than intimidating.

### 1.2 Non-Linear over Linear Functions

<a id="12-non-linear-over-linear-functions"></a>

#### 1.2.1 Neural network representation

<a id="121-neural-network-representation"></a>

**The idea in one line:** A neural network is logistic regression repeated many times in parallel and then stacked in layers.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Input layer** | Your features, `a[0] = X`. Conventionally *not* counted as a layer. |
| **Hidden layer** | Middle layers. "Hidden" because training data doesn't specify their values. |
| **Output layer** | The final layer producing ŷ. |
| **2-layer network** | 1 hidden layer + 1 output layer. The input layer isn't counted. |
| `a[l]` | Activations of layer l, the vector of outputs from that layer. |
| `a[l]_i` | The i-th neuron in layer l. |

**Deeper explanation:**

A network with 3 inputs, a 4-unit hidden layer, and 1 output. Each of the four hidden neurons does exactly what logistic regression does: takes all three inputs, computes `z = wᵀx + b`, applies an activation. They differ only in their weights, so they learn to detect different things.

The parameter shapes for this network:

- `W[1]`: (4, 3), four neurons, three inputs each
- `b[1]`: (4, 1)
- `W[2]`: (1, 4), one output neuron, four inputs
- `b[2]`: (1, 1)

**The general shape rule, which you should tattoo somewhere:**

```text
W[l] has shape (n[l], n[l-1])
b[l] has shape (n[l], 1)
```

where `n[l]` is the number of units in layer l. If your shapes match this rule, your forward pass will work. If they don't, it won't. This one rule catches most beginner bugs.

**Forward propagation for one example:**

```text
z[1] = W[1] a[0] + b[1]        a[0] = x
a[1] = g(z[1])
z[2] = W[2] a[1] + b[2]
a[2] = σ(z[2]) = ŷ
```

Four lines. That's a two-layer neural network.

**Real project example (CreatorRank):** Input: 20 creator metrics. Hidden layer: 12 units. Output: 1 (probability of campaign success). The 12 hidden units will spontaneously specialise, one might respond to "high engagement relative to follower count," another to "posting consistency," another to "audience geography match." You never named these. They emerged because they helped reduce the cost.

#### 1.2.2 Vectorizing across multiple examples

<a id="122-vectorizing-across-multiple-examples"></a>

**The idea in one line:** Stack examples as columns and the per-example equations become matrix equations, unchanged in form.

**Deeper explanation:**

Non-vectorized, you'd loop:

```text
for i in range(m):
    z1 = W1 @ x[i] + b1
    a1 = g(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
```

Vectorized, stack everything horizontally:

```text
Z1 = W1 @ X + b1        # X is (n_x, m) → Z1 is (n1, m)
A1 = g(Z1)
Z2 = W2 @ A1 + b2       # (n2, m)
A2 = sigmoid(Z2)
```

**How to read these matrices:** in `Z1` and `A1`, moving **horizontally across columns** means moving across training examples. Moving **vertically down rows** means moving across hidden units. Every column is one example's worth of activations.

The bias `b1` has shape (n1, 1) and broadcasts across all m columns automatically.

**Real project example:** This is why GPUs are transformative for deep learning. `W1 @ X` with a batch of 256 examples is one big matrix multiply, precisely the operation GPUs are built to do thousands of times in parallel. Larger batch sizes mean better GPU utilisation, up to the point where memory runs out.

#### 1.2.3 Activation functions

<a id="123-activation-functions"></a>

**The idea in one line:** The non-linear squashing function each neuron applies, and the choice matters more than beginners expect.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Activation function** `g(z)` | The non-linear function applied after the linear step. |
| **Saturation** | When the function flattens out and its slope goes to nearly zero, learning stalls there. |
| **Dead neuron** | A neuron whose output is always zero and whose gradient is always zero, so it never recovers. |

**Zero-centred**Outputs spread around zero rather than all positive. Helps optimisation.

Now the four you need to know.

#### 1.2.4 Sigmoid

<a id="124-sigmoid"></a>

```text
σ(z) = 1 / (1 + e^(−z))          range (0, 1)
σ'(z) = σ(z)(1 − σ(z))           max slope 0.25 at z = 0
```

**Use it:** for the **output layer of binary classification**, where you need a probability.

**Don't use it:** in hidden layers. Two reasons.

1. **Saturation.** For |z| > 5, the slope is essentially zero. Gradients that pass through get multiplied by ~0, so earlier layers learn almost nothing. This is the **vanishing gradient problem**, and it's why deep networks were nearly untrainable before ReLU.
2. **Not zero-centred.** All outputs are positive, which makes gradient updates zig-zag inefficiently.

**Memorable property:** the derivative is `a(1−a)` where `a` is the output you already computed. So in code you never recompute the exponential, you reuse the cached activation. Cheap.

#### 1.2.5 Tanh

<a id="125-tanh"></a>

```text
tanh(z) = (e^z − e^(−z)) / (e^z + e^(−z))      range (−1, 1)
tanh'(z) = 1 − tanh(z)²                        max slope 1 at z = 0
```

Tanh is a rescaled, shifted sigmoid, and it is **strictly better than sigmoid for hidden layers**, because it's zero-centred. Activations averaging around zero make the next layer's optimisation better-conditioned, roughly the same benefit you get from normalising inputs.

But tanh still saturates at both ends. So it still has vanishing-gradient problems in deep networks. It survives mainly inside LSTM/GRU cells, where the bounded (−1, 1) range is deliberately useful for gating.

**Rule of thumb:** tanh beats sigmoid everywhere except the binary output layer. But ReLU beats tanh in most hidden layers.

#### 1.2.6 ReLU and Leaky ReLU

<a id="126-relu-and-leaky-relu"></a>

```text
ReLU(z) = max(0, z)
ReLU'(z) = 0 if z < 0,  1 if z > 0   (undefined at 0; code just picks one)
LeakyReLU(z) = max(0.01z, z)
LeakyReLU'(z) = 0.01 if z < 0,  1 if z > 0
```

**ReLU is the default hidden-layer activation.** If you don't know what to use, use ReLU.

Why it works so well:

- **No saturation for positive z.** The slope is exactly 1, so gradients pass through undiminished. Deep networks actually train.
- **Cheap.** A comparison and a max. No exponentials.
- **Sparsity.** About half the neurons output exactly zero for any given input, which acts as a mild regulariser and makes computation more efficient.

The downside is **dying ReLU**: if a neuron's weights drift such that z is negative for every input in your dataset, its gradient is permanently zero and it can never recover. It's dead weight forever. In practice this affects a minority of neurons and rarely hurts much, because you have plenty of others, but it's real.

**Leaky ReLU** fixes it by giving a small slope (0.01) for negative z, so gradients never fully die. It often works slightly better than ReLU, yet ReLU remains more popular out of simplicity and inertia. Modern relatives worth knowing: **ELU**, **GELU** (standard in Transformers), and **Swish/SiLU**.

**The practical decision table:**

| **Situation** | **Use** |
| --- | --- |
| Hidden layers, default | **ReLU** |
| Hidden layers, seeing dead units or want a small edge | **Leaky ReLU** |
| Output, binary classification | **Sigmoid** |
| Output, multi-class (pick one of K) | **Softmax** |
| Output, multi-label (several can be true) | **Sigmoid on each output** |
| Output, regression (any real number) | **None (linear)** |
| Output, regression (non-negative, e.g. price) | **ReLU** |
| Inside LSTM/GRU gates | **Sigmoid + tanh** |

**Real project example (CatShop):** Predicting a product's weekly revenue. Hidden layers: ReLU. Output layer: **linear**: no activation at all, because revenue is an unbounded positive number, and forcing it through a sigmoid would cap it at 1. Beginners frequently put a sigmoid on a regression output and then wonder why every prediction is 0.99. Match the output activation to the *range* of the thing you're predicting.

#### 1.2.7 Why non-linear activation functions at all?

<a id="127-why-non-linear-activation-functions-at-all"></a>

**The idea in one line:** Without a non-linearity, a hundred-layer network collapses algebraically into a single linear layer, so all that depth buys you nothing.

**Deeper explanation:**

Suppose you use the identity activation, `g(z) = z`. Then:

```text
a[1] = W[1]x + b[1]
a[2] = W[2]a[1] + b[2]
```

```text
     = W[2](W[1]x + b[1]) + b[2]
     = (W[2]W[1])x + (W[2]b[1] + b[2])
     = W' x + b'
```

Two layers became one. The composition of linear functions is linear. Add fifty more layers and you *still* have a single linear function, one with an enormous parameter count and exactly the expressive power of plain linear regression. It could never learn XOR, let alone recognise a cat.

**The non-linearity is what makes depth meaningful.** Each layer bends the space; stacking bends produce arbitrarily complex decision boundaries.

**The one legitimate exception:** the **output layer of a regression problem** should be linear, because the output genuinely can be any real number. Hidden layers must always be non-linear. (A rarer exception: some compression/bottleneck architectures deliberately use a linear layer to reduce dimensionality.)

**Real project example:** If you ever build a deep network and its accuracy matches your linear baseline exactly, check your activations first. A missing `activation='relu'` in a Keras `Dense` layer defaults to linear and silently turns your deep model into linear regression. No error, no warning, just mediocre results.

#### 1.2.8 Gradient descent for neural networks

<a id="128-gradient-descent-for-neural-networks"></a>

**The idea in one line:** Same loop as logistic regression, forward, compute cost, backward, update, just with more layers of parameters.

**Deeper explanation:**

Parameters: `W[1], b[1], W[2], b[2]`. Cost: the same cross-entropy average.

```text
repeat:
    compute predictions Ŷ = A[2]     (forward propagation)
    compute dW[1], db[1], dW[2], db[2]   (backward propagation)
    W[1] := W[1] − α·dW[1]
    b[1] := b[1] − α·db[1]
    W[2] := W[2] − α·dW[2]
    b[2] := b[2] − α·db[2]
```

**Forward propagation (vectorized):**

```text
Z[1] = W[1]X  + b[1]
A[1] = g[1](Z[1])
Z[2] = W[2]A[1] + b[2]
A[2] = σ(Z[2])
```

**Backward propagation (vectorized):**

```python
dZ[2] = A[2] − Y
dW[2] = (1/m) · dZ[2] · A[1]ᵀ
db[2] = (1/m) · np.sum(dZ[2], axis=1, keepdims=True)
```

```python
dZ[1] = W[2]ᵀ · dZ[2] * g[1]'(Z[1])        ← * is ELEMENTWISE multiply
dW[1] = (1/m) · dZ[1] · Xᵀ
db[1] = (1/m) · np.sum(dZ[1], axis=1, keepdims=True)
```

**The one line that carries all the meaning** is `dZ[1] = W[2]ᵀ` `dZ[2] * g'(Z[1])`. Read it as two steps:

1. `W[2]ᵀ` `dZ[2]`, take the error signal at layer 2 and route it backward through the weights, so each layer-1 neuron receives blame proportional to how much it influenced layer 2.
2. `* g'(Z[1])`, scale that blame by the local slope of the activation. **If the activation was saturated (slope ≈ 0), the blame gets multiplied by nearly zero and the neuron learns nothing.** This is exactly the vanishing gradient problem, visible in a single symbol.

Note `keepdims=True` in the sum. Without it numpy returns a rank-1 array of shape (n,) instead of (n,1), and later broadcasting silently does something wrong. Use it every time.

#### 1.2.9 Derivative formulas

<a id="129-derivative-formulas"></a>

Reference table. Each derivative is expressible in terms of the activation `a` you already computed, so nothing needs recomputing.

| **Activation** | **g(z)** | **g'(z)** |
| --- | --- | --- |
| Sigmoid | `1/(1+e^-z)` | `a(1 − a)` |
| Tanh | `tanh(z)` | `1 − a²` |
| ReLU | `max(0, z)` | `0 if z<0 else 1` |
| Leaky ReLU | `max(0.01z, z)` | `0.01 if z<0 else 1` |
| Linear | `z` | `1` |

**Why sigmoid causes vanishing gradients, numerically:** the maximum of `a(1−a)` is 0.25, at a = 0.5. So *every* sigmoid layer multiplies the backward signal by at most 0.25. Ten sigmoid layers → at best 0.25¹⁰ ≈

0. 000001. The first layer receives a gradient a million times smaller than the last. It effectively doesn't train.

ReLU's derivative is exactly 1 for positive inputs, so the signal passes through intact. That single fact is a large part of why deep learning works.

#### 1.2.10 Zero initialization, and why it fails

<a id="1210-zero-initialization-and-why-it-fails"></a>

**The idea in one line:** If all weights start at zero, every neuron in a layer computes the identical thing forever, and your 100-unit layer has the power of 1 unit.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Symmetry breaking** | Making neurons different from each other so they can specialise. |
| **Symmetric / identical units** | Neurons computing exactly the same function, wasted capacity. |

**Deeper explanation:**

Set `W[1] = np.zeros((4, 3))`. Then in the forward pass, all four hidden neurons compute the same z, so the same a. In the backward pass, they all receive the same gradient. So they all update identically. After a thousand iterations they are *still* identical.

By induction, they remain identical forever. Your four-unit layer computes one function, four times. Adding more units changes nothing.

**Important nuance: biases are fine at zero.**`b = np.zeros((n, 1))` is standard. Symmetry is broken by the *weights*; once the weights differ, the neurons diverge regardless of the biases.

#### 1.2.11 Random initialization

<a id="1211-random-initialization"></a>

**The idea in one line:** Start weights small and random, random enough to break symmetry, small enough to avoid saturation.

**Deeper explanation:**

The classic starting point:

```python
W1 = np.random.randn(n1, n0) * 0.01
b1 = np.zeros((n1, 1))
```

Two design choices:

**Why random?** Symmetry breaking, per above.

**Why multiply by 0.01 (i.e. why small)?** If weights are large, z is large, and with sigmoid/tanh you start deep in the saturated flat region where the slope is near zero. Learning crawls from step one. Small weights keep z near zero where the slope is steepest.

**But 0.01 is not the right answer for deep networks.** A fixed small constant makes activations shrink layer by layer in a deep net until they vanish. The modern approach scales the initialisation by the number of inputs to each layer:

```python
# He initialization - the default for ReLU
W = np.random.randn(n_l, n_prev) * np.sqrt(2.0 / n_prev)
# Xavier / Glorot initialization - for tanh / sigmoid
W = np.random.randn(n_l, n_prev) * np.sqrt(1.0 / n_prev)
```

The reasoning: a neuron with 500 inputs sums 500 terms, so each term must be smaller to keep the total in a sensible range. Dividing the variance by `n_prev` keeps the scale of activations roughly constant as you go deeper. This is one of the changes that made very deep networks trainable, and we'll return to it under vanishing/exploding gradients in Chapter 2.

**Real project example:** In Keras this is handled for you, `Dense(64, activation='relu')` uses

Glorot uniform by default, and you can set `kernel_initializer='he_normal'` for ReLU layers. Doing so on a 20-layer network can be the difference between converging in 10 epochs and not converging at all. Free performance, one argument.

### 1.3 Deep Neural Networks

<a id="13-deep-neural-networks"></a>

#### 1.3.1 What is a deep neural network?

<a id="131-what-is-a-deep-neural-network"></a>

**The idea in one line:** Just a network with several hidden layers, "deep" is a relative, not a technical, term.

**Deeper explanation:**

Logistic regression is a 1-layer network (the shallowest possible). Two or three hidden layers is a small network. Modern vision networks run 50–150 layers; large language models are deeper still.

Layer count is a **hyperparameter** you tune empirically. There's no formula. Start shallow, add depth while the dev-set error keeps improving.

**Real project example:** For structured business data (churn, lead scoring, pricing) 2–4 hidden layers is almost always enough, and often 1 is. Depth pays off most for perceptual data, images, audio, language, where there's a genuine hierarchy of features to discover. Don't build a 20-layer network for a 30-column CSV.

#### 1.3.2 Deep neural network notation

<a id="132-deep-neural-network-notation"></a>

| **Symbol** | **Meaning** |
| --- | --- |
| `L` | Total number of layers (not counting input) |
| `n[l]` | Number of units in layer l |
| `n[0] = n_x` | Number of input features |
| `a[l]` | Activations of layer l |
| `a[0] = X` | The input |
| `a[L] = ŷ` | The prediction |

`W[l]`, Parameters of layer l

```python
 b[l]
```

`g[l]`Activation function of layer l

**The shape rules, the most useful debugging tool in deep learning:**

```text
W[l]  : (n[l], n[l-1])
b[l]  : (n[l], 1)
Z[l]  : (n[l], m)
A[l]  : (n[l], m)
dW[l] : same shape as W[l]
db[l] : same shape as b[l]
```

`dW` always has the same shape as `W`. If it doesn't, you have a bug. This is the single fastest way to catch errors in hand-written backprop.

#### 1.3.3 Forward propagation in a deep network

<a id="133-forward-propagation-in-a-deep-network"></a>

**The general rule for any layer l:**

```text
Z[l] = W[l] · A[l−1] + b[l]
A[l] = g[l](Z[l])
```

with `A[0] = X`, applied for l = 1 to L.

```text
A = X
for l in range(1, L+1):
    Z = W[l] @ A + b[l]
    A = relu(Z) if l < L else sigmoid(Z)
```

This is the one place where an explicit `for` loop is not only acceptable but necessary, layers are genuinely sequential; layer 3 can't start until layer 2 is done. Vectorize across *examples*, loop over *layers*.

You must also **cache**`Z[l]` (and `A[l−1]`, `W[l]`) during the forward pass, because backprop needs them. This is why training uses much more memory than inference: inference can discard each layer's intermediates immediately, training cannot.

#### 1.3.4 Forward and backward functions

<a id="134-forward-and-backward-functions"></a>

**The idea in one line:** Think of each layer as a block with a forward function and a backward function, wired into a chain.

**Deeper explanation:**

FORWARD, layer l:

```text
  input:  A[l−1]
  output: A[l]
  cache:  Z[l], W[l], b[l], A[l−1]
BACKWARD, layer l:
  input:  dA[l]  (+ the cache)
  output: dA[l−1], dW[l], db[l]
```

The full training step:

```text
X → [fwd 1] → [fwd 2] → ... → [fwd L] → ŷ → compute cost
                                              ↓
dW,db ← [bwd 1] ← [bwd 2] ← ... ← [bwd L] ← dA[L]
```

with `dA[L] = −y/a + (1−y)/(1−a)` starting the chain.

**This modular view is exactly how PyTorch and TensorFlow are architected.** Every layer type, Dense, Conv2D, LSTM, BatchNorm, Dropout, is a class implementing forward and backward. That's why you can stack arbitrary layers in any order and the framework just works: each block only needs to know how to pass a signal forward and a gradient backward. Understanding this makes writing custom layers unintimidating.

#### 1.3.5 Forward propagation for layer l

<a id="135-forward-propagation-for-layer-l"></a>

```text
Z[l] = W[l] A[l−1] + b[l]
A[l] = g[l](Z[l])
cache: Z[l]
```

Sanity-check the shapes with a concrete example. Network `[n0=12288, n1=500, n2=100, n3=1]`, batch of m=64:

|   | **Shape** |
| --- | --- |
| `X = A[0]` | (12288, 64) |
| `W[1]` | (500, 12288) |
| `Z[1] = W[1]A[0] + b[1]` | (500, 64) |
| `A[1]` | (500, 64) |
| `W[2]` | (100, 500) |
| `A[2]` | (100, 64) |
| `W[3]` | (1, 100) |
| `A[3] = Ŷ` | (1, 64) |

Note `W[1]` alone has 500 × 12288 = 6.1 million parameters. This is exactly why images use CNNs, a convolutional layer with 32 filters of size 3×3×3 has only 896 parameters and works far better, because it exploits the spatial structure a dense layer throws away.

#### 1.3.6 Backward propagation for layer l

<a id="136-backward-propagation-for-layer-l"></a>

```python
dZ[l] = dA[l] * g[l]'(Z[l])
dW[l] = (1/m) · dZ[l] · A[l−1]ᵀ
db[l] = (1/m) · np.sum(dZ[l], axis=1, keepdims=True)
dA[l−1] = W[l]ᵀ · dZ[l]
```

Read the four lines as a story:

1. Convert "how the *output* should change" into "how the *pre-activation* should change" by multiplying by the local slope.
2. Weight gradient = error signal × the input that flowed in. Big input × big error = big update.
3. Bias gradient = the error signal, summed across the batch.
4. Pass the blame back to the previous layer through the transposed weights.

For the final layer with sigmoid output and cross-entropy loss, all of this collapses to `dZ[L] = A[L] − Y`, exactly as in logistic regression.

**Real project example:** In practice `loss.backward()` in PyTorch does all of this. But when you get

`RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x500 and 100x64)`, the shape rules above tell you immediately which layer is misconfigured and in which direction. That's the payoff for learning this.

#### 1.3.7 Parameters vs. hyperparameters

<a id="137-parameters-vs-hyperparameters"></a>

**The idea in one line:** Parameters are learned by the algorithm; hyperparameters are chosen by you, and choosing them well is most of the job.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Parameters** | `W` and `b`. Learned automatically by gradient descent. |
| **Hyperparameters** | Settings you pick *before* training that control *how* learning happens. |

**The main hyperparameters:**

**HyperparameterWhat it controlsTypical values**

Learning rate α Step size 0.0001 – 0.1 (log scale)

| Number of iterations / epochs | How long to train | 10 – 1000+ |
| --- | --- | --- |
| Number of hidden layers L | Depth | 1 – 100+ |
| Units per layer n[l] | Width | 8 – 4096 |
| Activation function | Non-linearity | ReLU by default |
| Mini-batch size | Examples per update | 32, 64, 128, 256, 512 |
| Regularization strength λ | Overfitting control | 0.0001 – 10 |
| Dropout rate | Overfitting control | 0.1 – 0.5 |
| Momentum β / Adam β₁, β₂ | Optimizer behaviour | 0.9, 0.999 |

**Learning rate is the most important by a wide margin.** Get it wrong and nothing else matters. Tune it first, always.

**Real project example:** Keep a spreadsheet or an experiment tracker (Weights & Biases, MLflow, or a plain CSV) with one row per run: every hyperparameter, plus train and dev error. After thirty runs you'll see patterns that no amount of theorising would have given you. Untracked experiments are wasted experiments, you *will* forget which run produced the good number.

#### 1.3.8 Applied deep learning is an empirical process

<a id="138-applied-deep-learning-is-an-empirical-process"></a>

**The idea in one line:** Nobody knows the right hyperparameters in advance. You cycle Idea → Code → Experiment, and the winner is whoever cycles fastest.

**Deeper explanation:**

There is no formula that outputs the correct architecture for your problem. Practitioners with a decade of experience still guess, run, and adjust. What experience buys you is a better *initial* guess and a faster read on what the results mean, not the ability to skip the loop.

Practical implications:

- **Optimise iteration speed above almost everything.** Train on a 10% subsample while debugging. Get a bad model working end-to-end before you try to make it good.
- **Intuitions don't transfer cleanly across domains.** Settings that work for NLP often fail for vision or structured data. Re-tune when you switch domains.
- **Re-tune periodically.** Data distributions drift; hardware changes; what was optimal six months ago may not be now.
- **Change one thing at a time**, or you won't know what caused the improvement.

**Real project example (SaaSChurn), a realistic first week:**

1. Day 1: Logistic regression baseline. AUC 0.71. *Now you have a number to beat.*
2. Day 2: One hidden layer, 32 units. AUC 0.73. Slight gain.
3. Day 2: Learning rate sweep {0.0001, 0.001, 0.01}. Best is 0.001. AUC 0.75.
4. Day 3: Two hidden layers. Train AUC 0.95, dev AUC 0.74 → **overfitting**, diagnosed by the gap.
5. Day 3: Add dropout 0.3. Dev AUC 0.78.
6. Day 4: Add three engineered features (usage trend slope, seat utilisation ratio, ticket sentiment). Dev AUC 0.83. **The biggest single win came from features, not architecture.**
7. Day 5: Error analysis on the 100 worst misses → discover enterprise accounts behave differently → train a separate model for them.

That's what the work actually looks like. Notice how much of it is diagnosis rather than modelling, which is exactly what Chapter 2 and Chapter 4 are about.

## Chapter 2: Optimization Strategies in Neural Networks

<a id="chapter-2-optimization-strategies-in-neural-networks"></a>

### 2.1 How to Improve Neural Networks

<a id="21-how-to-improve-neural-networks"></a>

#### 2.1.1 Train / dev / test sets

<a id="211-train-dev-test-sets"></a>

**The idea in one line:** Split your data three ways, one part to learn from, one to make decisions with, one to get an honest final number.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Training set** | Data the model learns its weights from. |
| **Dev set** (a.k.a. validation set, hold-out set) | Data used to compare models and tune hyperparameters. Not trained on. |
| **Test set** | Touched once, at the very end, for an unbiased performance estimate. |
| **Hold-out cross-validation** | Another name for the dev set. |

**Deeper explanation:**

Why three and not two? Because **anything you tune against, you overfit to.** If you try 200 hyperparameter combinations and pick the best on the dev set, that best score is optimistically biased, you've partly fit the dev set's noise. The test set, never used for any decision, gives you the number you can actually report.

**Sizing:**

**Dataset Split**

**size**

1,000 60 / 20 / 20

| 10,000 | 60 / 20 / 20 |
| --- | --- |
| 1,000,000 | 98 / 1 / 1 |
| 10,000,000 | 99.5 / 0.4 / 0.1 |

The old 70/30 and 60/20/20 rules come from an era of small datasets. The right question is not "what percentage?" but **"how many examples do I need for a reliable estimate?"** Ten thousand dev examples give you a very tight confidence interval; there's no reason to spend 200,000 on it when you could be training on them instead.

Some teams skip the test set and call the dev set "test." It's not recommended, you lose your unbiased estimate, but it's common and not fatal if you understand the bias you're accepting.

**Real project example (SaaSChurn):** Time matters here. Don't split randomly. Train on Jan–Sep, dev on Oct, test on Nov–Dec. A random split lets the model see the future, it might learn from an account's December behaviour to predict its October churn, giving you a beautiful offline score and a useless production model. **For any time-dependent problem, split by time.** This mistake, called leakage, is probably the most common way real ML projects silently fail.

#### 2.1.2 Mismatched train/test distribution

<a id="212-mismatched-traintest-distribution"></a>

**The idea in one line:** If your training data doesn't look like your production data, your metrics are lying to you.

**Deeper explanation:**

Classic case: you train a cat classifier on 200,000 crisp images scraped from the web, and deploy it in an app where users upload blurry, badly lit phone photos. Training accuracy is superb. Production accuracy is terrible. Nothing is broken, you simply optimised for the wrong distribution.

**The rule: dev and test sets must come from the distribution you actually care about.** Training data can come from anywhere. The dev/test sets define your target, so they must reflect reality.

A concrete allocation for the cat app: train on 200k web images plus 5k user photos; make the dev and test sets *purely* user photos. You now have a target that matches the mission, even though most of the training signal comes from elsewhere.

**Dev and test must always come from the same distribution as each other.** If dev is US users and test is Indian users, you're optimising toward one target and measuring against another, aiming at a moving target. Pool them and shuffle.

**Real project example (CreatorRank):** You bootstrap training labels with a heuristic on 100,000 creators. But your dev and test sets should be a few thousand creators whose campaigns were *actually run and measured*, with real ROI. Cheap proxy labels for training, expensive real labels for evaluation. This is a very common and very sensible pattern.

#### 2.1.3 Bias and variance

<a id="213-bias-and-variance"></a>

**The idea in one line:** Bias is underfitting (too simple to capture the pattern); variance is overfitting (memorising noise instead of learning the pattern).

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Bias** | Model too simple. Bad on training data *and* dev data. |
| **Variance** | Model too complex. Great on training data, bad on dev data. |
| **Underfitting** | High bias. |
| **Overfitting** | High variance. |
| **Bayes error / optimal error** | The best error any possible model could achieve, irreducible noise in the problem. |

**Deeper explanation, the diagnostic table:**

Assume Bayes error ≈ 0% (e.g. humans classify cats near-perfectly).

| **Train error** | **Dev error** | **Diagnosis** |
| --- | --- | --- |
| 1% | 11% | **High variance**: memorising the training set |
| 15% | 16% | **High bias**: can't even fit the training data |
| 15% | 30% | **High bias AND high variance**: worst case |
| 0.5% | 1% | **Low bias, low variance**: good |

The logic:

- **Train error vs. Bayes error → bias.** How far is the model from the best possible on data it has literally seen?
- **Dev error vs. train error → variance.** How much worse does it get on unseen data?

**The Bayes error caveat matters enormously.** If your task is blurry image classification where even expert humans get 15% wrong, then 15% train / 16% dev is not high bias, it's an excellent model near the ceiling. **Always establish a human-level baseline before diagnosing bias.** We'll go much deeper on this in Chapter 4.

**Real project example (SaaSChurn):** Train AUC 0.95, dev AUC 0.74 → clear high variance. Don't add layers; add regularisation or data. Train AUC 0.72, dev AUC 0.71 → high bias. More data won't help at all, you need a bigger model or better features. **Getting this diagnosis right determines whether your next two weeks are productive or wasted.** It's the single highest-leverage habit in applied ML.

#### 2.1.4 The basic recipe for machine learning

<a id="214-the-basic-recipe-for-machine-learning"></a>

**The idea in one line:** A decision tree you should run after every single training run.

Train a model.

```text
┌─ Is training error high? (compare to human/Bayes level)
│    YES → HIGH BIAS. Do:
│           • Bigger network (more layers, more units)
│           • Train longer
│           • Better optimizer (Adam)
│           • Different architecture
│           • Better features
│         → repeat until training error is acceptable
│
└─ NO ↓
┌─ Is dev error much worse than training error?
│    YES → HIGH VARIANCE. Do:
│           • Get more training data  ← most reliable fix
│           • Regularization (L2, dropout)
│           • Data augmentation
│           • Different architecture
│         → repeat
│
└─ NO → Done. Ship it.
```

**Two things to notice:**

1. **The order is not negotiable.** Fix bias first. There is no point regularising a model that can't even fit its training data, you'll only make it worse.
2. **The "bias-variance tradeoff" is much weaker in deep learning than in classical ML.** In the classical framing, reducing one increases the other. But in deep learning you have two tools that break the trade: a **bigger network** reduces bias without necessarily increasing variance (given regularisation), and **more data** reduces variance without increasing bias. That's why "get more data and train a bigger network" is such a durable strategy.

**Real project example (CatShop):** Products misclassified into wrong categories. Train error 12%, dev error 13%, human error 2%. That's a bias problem, the diagnosis is unambiguous. Collecting 500,000 more product images would be an expensive waste. Instead: go deeper, use a pretrained ResNet backbone, train longer. Six weeks and a large data-collection budget saved by one comparison of three numbers.

#### 2.1.5 Regularization (logistic regression)

<a id="215-regularization-logistic-regression"></a>

**The idea in one line:** Add a penalty for large weights, so the model prefers simple explanations.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Regularization** | Anything that reduces overfitting. |
| **L2 regularization** | Penalise the sum of squared weights. The default. |
| **L1 regularization** | Penalise the sum of absolute weights. Drives some weights to exactly zero → sparsity. |
| **λ (lambda)** | Regularization strength, a hyperparameter. |

**Deeper explanation:**

```text
J(w,b) = (1/m)·Σ L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)  +  (λ/2m)·||w||²
         └──── fit the data ────┘   └── stay simple ──┘
```

where `||w||² = Σ w²`.

Now the optimiser has two competing pressures. It wants to fit the data, but every large weight costs it. It will only make a weight large if the accuracy gain justifies the penalty. The result is that spurious patterns, which give small gains, get suppressed while genuine patterns survive.

- **λ = 0** → no regularization → maximum overfitting risk.
- **λ very large** → weights forced toward zero → model becomes nearly a constant → underfitting.
- **λ tuned on the dev set** → the sweet spot.

Note that `b` is not usually regularized. It's a single parameter and contributes negligibly to overfitting.

**In code:**`lambd`, because `lambda` is a reserved keyword in Python.

**L1 vs L2 in practice:** L1 gives sparse solutions (many weights exactly zero), which is useful for feature selection. L2 is the standard for neural networks and is what almost everyone means by "regularization."

**Real project example (SaaSChurn):** With 200 features and 4,000 accounts, you will overfit. L2 with λ tuned across {0.001, 0.01, 0.1, 1, 10} typically finds a sweet spot around 0.1. If you also want the model to *tell you which features matter*, use L1, it will zero out the irrelevant ones, which is a genuinely useful output for your product team, not just a modelling trick.

#### 2.1.6 Regularization (neural networks)

<a id="216-regularization-neural-networks"></a>

**The idea in one line:** Same principle, summed over every weight matrix in the network.

```text
J = (1/m)·Σ L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾) + (λ/2m)·Σ_l ||W[l]||²_F
```

**Jargon decoded:**`||W||²_F` is the **Frobenius norm** squared, sum of the squares of every element in the matrix. It's called "Frobenius" rather than "L2" for historical reasons in linear algebra; conceptually it's the same thing applied to a matrix.

**Effect on the gradient:**

```text
dW[l] = (backprop term) + (λ/m)·W[l]
W[l] := W[l] − α·dW[l]
```

Substituting and rearranging:

W[l]:= (1 − αλ/m)·W[l] − α·(backprop term)

That factor `(1 − αλ/m)` is slightly less than 1, so **every single update shrinks all the weights a little before applying the gradient.** Hence L2 regularization's other name: **weight decay**.

**Real project example:** In Keras: `Dense(64, activation='relu',`

```text
kernel_regularizer=regularizers.l2(0.01)). In PyTorch, optim.Adam(params,
```

`weight_decay=0.01)`. Note the subtlety that Adam's naive weight decay is mathematically not quite the same as L2 penalty, which is why **AdamW** exists and is now the recommended optimizer for most modern work.

#### 2.1.7 How does regularization prevent overfitting?

<a id="217-how-does-regularization-prevent-overfitting"></a>

Two complementary intuitions.

**Intuition 1, shrinking toward a simpler network.** With λ very large, the penalty dominates and weights go near zero. A neuron with near-zero weights outputs near-zero regardless of input, effectively removed. So the network behaves like a much smaller one, which can't overfit. Intermediate λ gives you a network that's *effectively* smaller than its parameter count suggests. You're smoothly interpolating between "huge network" and "tiny network" with a continuous dial.

**Intuition 2, staying in the linear region.** With tanh activations: small weights → small z → and for small z, tanh is nearly *linear*. So every layer behaves nearly linearly, and (per section 1.2.7) stacked linear layers give a nearly linear model, which can only draw simple decision boundaries. Large λ pushes the network toward simple, smooth boundaries; small λ frees it to draw wiggly ones that curve around individual training points.

**A debugging note:** when you add regularization, your plotted cost function must include the regularization term. Otherwise the curve may not decrease monotonically and you'll think something is broken when it isn't.

#### 2.1.8 Dropout regularization

<a id="218-dropout-regularization"></a>

**The idea in one line:** Randomly switch off a fraction of neurons on every training step, so the network can't rely on any single one.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Dropout** | Randomly zeroing out units during training. |
| **keep_prob** | Probability of *keeping* a unit. keep_prob = 0.8 means drop 20%. |
| **dropout rate** | Probability of *dropping*. rate = 0.2 means drop 20%. (Keras uses this convention, opposite of keep_prob. Read the docs.) |
| **Inverted dropout** | The standard implementation, which rescales during training so test time needs no change. |

**Deeper explanation:**

On each training example (and each iteration), independently decide for each hidden unit whether to keep it. Dropped units output zero for that pass, both forward and backward.

The effect is that the network sees a different, thinner architecture every step. It can never rely on one specific neuron being present, so it **spreads its bets**: distributing importance across many units rather than concentrating on a few. Spreading weights out has an effect mathematically similar to L2 regularization.

Another framing: dropout trains an enormous **ensemble** of subnetworks that share weights, and averages them at test time. Ensembles reduce variance; that's exactly what we want.

#### 2.1.9 Implementing dropout (inverted dropout)

<a id="219-implementing-dropout-inverted-dropout"></a>

```python
# Forward pass, layer 3, keep_prob = 0.8
d3 = np.random.rand(*a3.shape) < keep_prob   # boolean mask, ~80% True
a3 = a3 * d3                                  # zero out ~20% of activations
a3 = a3 / keep_prob                           # ← the "inverted" step
```

**Why divide by keep_prob?** Zeroing 20% of activations reduces the expected value of `z` in the next layer by 20%. Dividing by 0.8 scales it back up, so the *expected* magnitude of the activations is unchanged. That means test time, where you use the full network, needs no adjustment at all. Without this step, activations at test time would be systematically ~25% larger than what the next layer was trained to expect, and everything would be miscalibrated.

**Critical details:**

1. **Use a different random mask each iteration**, not a fixed one. The randomness is the whole point.
2. **The same** `d3` **mask must be used in the backward pass** for that layer, so gradients don't flow through dropped units.

#### 2.1.10 Making predictions at test time

<a id="2110-making-predictions-at-test-time"></a>

**The rule: no dropout at test time.** Use the full network with all units active, and no scaling (thanks to inverted dropout).

Reasons:

- You want deterministic predictions. The same input should always give the same output.
- You want maximum accuracy; there's no reason to cripple the network at inference.

**Real project example, a bug you will encounter:** In PyTorch you must call `model.eval()` before

inference and `model.train()` before training. Forgetting `model.eval()` means dropout stays active in production, so predictions are random and worse. It's silent, no error, no warning, just degraded accuracy that's hard to trace. (`model.eval()` also switches BatchNorm to its inference behaviour, which we'll cover

in 2.3.) Keras handles this automatically when you call `.predict()`.

#### 2.1.11 Why does dropout work?

<a id="2111-why-does-dropout-work"></a>

**The core intuition:***A neuron cannot rely on any one input, because that input might vanish. So it must spread its weights across many inputs.* Spread-out weights are smaller weights, which is L2's effect achieved by a different route.

**Practical guidance:**

- **Vary keep_prob per layer.** Big layers with many parameters overfit most, so drop more aggressively there (keep_prob 0.5). Small layers can keep more (0.8–1.0). The input layer usually gets keep_prob near 1.0.
- **Dropout is standard in computer vision** because image data is high-dimensional and training data is almost always insufficient relative to model size.
- **Only use it if you're actually overfitting.** Dropout is a cure, not a vitamin. If train and dev error are close, dropout will just add bias and slow you down.
- **The downside:** the cost function is no longer well-defined (it changes every iteration), so you can't use the "is J decreasing monotonically?" debugging check. Standard workaround: turn dropout off (keep_prob = 1.0), verify the cost decreases cleanly, then turn it back on.

**Real project example (CatShop):** A product-image classifier with 8,000 training images. Train accuracy 99%, dev accuracy 76%, severe overfitting. Adding dropout 0.5 to the two dense layers before the output brings dev accuracy to 84%. Combining it with data augmentation gets you to 89%. Neither alone was sufficient; **regularization techniques stack**.

#### 2.1.12 Data augmentation

<a id="2112-data-augmentation"></a>

**The idea in one line:** Manufacture more training data by transforming what you have in ways that don't change the label.

**Deeper explanation:**

A horizontally flipped cat is still a cat. A slightly rotated, cropped, brightened, or zoomed cat is still a cat. So you can multiply your effective dataset size for free.

Common image augmentations: horizontal flip, random crop, rotation (±15°), brightness/contrast/saturation jitter, small random distortions, and **cutout** (masking a random patch).

**The critical rule: the transform must preserve the label.** Vertical flipping is fine for satellite imagery and disastrous for digit recognition (a flipped 6 is not a 6). Horizontal flipping is fine for cats and wrong for text.

It's cheaper than collecting real data, but weaker, the augmented examples are correlated with their originals, so 10× augmentation is not worth 10× real data. It's still one of the highest-ROI things you can do.

**Beyond images:**

- **Audio:** add background noise, shift pitch, change speed, time-mask.
- **Text:** synonym replacement, back-translation (English → French → English), random word deletion.
- **Tabular:** SMOTE for minority-class oversampling; small Gaussian noise on continuous features.

**Real project example (CreatorRank):** For matching creator aesthetics to brand guidelines, augment creator photos with colour jitter and random crops. But *don't* augment with heavy colour shifts if the brand aesthetic **is** about colour palette, you'd be destroying the very signal you're trying to learn. Augmentation choices must respect what the label actually depends on.

#### 2.1.13 Early stopping

<a id="2113-early-stopping"></a>

**The idea in one line:** Plot train and dev error against training time, and stop at the point where dev error is lowest.

**Deeper explanation:**

Training error decreases monotonically. Dev error decreases, bottoms out, then rises as the model begins overfitting. Early stopping halts at the bottom of that U.

*Why it acts as regularization:* weights start small (random init) and grow during training. Stopping early leaves them at a mid-sized value, which is roughly what L2 aims for. You're getting L2's effect via the time dimension.

**The main criticism (and it's a good one): early stopping is not orthogonal.** A single knob is simultaneously controlling how well you fit the training set (bias) and how much you overfit (variance). That violates the principle of one-knob-one-job, which we'll formalise in Chapter 4. The recommended alternative is to train as long as you like and control overfitting with L2 and dropout, tuning λ independently. The counter-argument is practical: early stopping is free, requires no extra hyperparameter search, and works. Most production teams use it, usually as `EarlyStopping(patience=10,`

`restore_best_weights=True)` in Keras, while also using dropout and L2.

**Real project example:** Always use `restore_best_weights=True`. Without it, you keep the weights

from the *last* epoch, which by definition is `patience` epochs past the best one, and therefore worse. Easy to miss, easy to fix.

#### 2.1.14 Normalizing training sets

<a id="2114-normalizing-training-sets"></a>

**The idea in one line:** Rescale every feature to have mean 0 and variance 1, so no feature dominates just because of its units.

**The two steps:**

```python
mu    = np.mean(X, axis=1, keepdims=True)
X     = X - mu                              # 1. zero the mean
sigma2 = np.mean(X**2, axis=1, keepdims=True)
X      = X / np.sqrt(sigma2 + 1e-8)         # 2. unit variance
```

**The rule you must not break: compute** `mu` **and** `sigma` **on the TRAINING set only, then apply those same values to dev, test, and production.** If you normalise the test set using its own statistics, you've leaked information about the test distribution into your pipeline, and your offline number will be optimistic. In scikit-learn: `scaler.fit_transform(X_train)` then `scaler.transform(X_test)`, never

`fit_transform` on test.

#### 2.1.15 Why normalize inputs?

<a id="2115-why-normalize-inputs"></a>

**The idea in one line:** Unnormalized features make the cost surface a long thin valley; normalized features make it a round bowl, and gradient descent handles bowls far better.

**Deeper explanation:**

Suppose feature x₁ ranges 0–1 (conversion rate) and x₂ ranges 0–1,000,000 (follower count). The cost function is then extremely elongated, very sensitive to w₂, barely sensitive to w₁. Contour lines form a narrow ravine.

Gradient descent on a ravine oscillates violently across the narrow direction while creeping slowly along the long one. You're forced to use a tiny learning rate to avoid divergence, so training is slow.

Normalized features give roughly circular contours. The gradient points more or less straight at the minimum from anywhere, so you can use a large learning rate and converge in far fewer steps.

**When it matters most:** when features have wildly different scales. If everything is already 0–1 (like pixel values divided by 255), the benefit is smaller, but normalising anyway never hurts, so just always do it.

**Real project example (CreatorRank):** Features include follower count (1,000 – 50,000,000), engagement rate (0.001 – 0.15), and average views (100 – 10,000,000). Without normalisation, training will barely move. Additionally, for heavily skewed count features like followers, apply `log1p`*before* standardising, the distribution is roughly log-normal, and log-transforming makes it approximately Gaussian, which the network handles far better. This kind of thoughtful preprocessing routinely beats architecture tweaking.

#### 2.1.16 Vanishing and exploding gradients

<a id="2116-vanishing-and-exploding-gradients"></a>

**The idea in one line:** In deep networks, gradients get multiplied through many layers, so they shrink toward zero or blow up toward infinity, and either way training breaks.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Vanishing gradients** | Gradients shrink exponentially with depth; early layers barely learn. |
| **Exploding gradients** | Gradients grow exponentially; weights update wildly, cost becomes NaN. |
| **Gradient clipping** | Capping gradient magnitude to prevent explosion. |

**Deeper explanation:**

Consider a very deep linear network (activations = identity, biases = 0). Then:

```text
ŷ = W[L]·W[L−1]·...·W[2]·W[1]·x
```

If every W is `1.5·I`, then ŷ ≈ 1.5^L · x. With L = 150, that's astronomically large, **exploding**.

If every W is `0.5·I`, then ŷ ≈ 0.5^150 · x ≈ 0, **vanishing**.

The same exponential behaviour applies to gradients on the way back. In a 150-layer network, a per-layer factor of 0.9 gives 0.9^150 ≈ 10⁻⁷, the first layer receives essentially no signal and never learns.

**The fixes, in rough order of importance:**

1. **Careful weight initialization.** Scale initial variance by the fan-in so activations neither grow nor shrink per layer. He init for ReLU, Xavier for tanh (section 1.2.11). This is the primary fix.
2. **ReLU instead of sigmoid/tanh.** Derivative of exactly 1 for positive inputs means no shrinkage per layer.
3. **Batch normalization** (section 2.3). Re-normalises activations at every layer, preventing drift.
4. **Residual/skip connections** (ResNets). Add a shortcut path `a[l+2] = g(z[l+2] + a[l])` so gradients have a direct highway back. This is the reason 150-layer networks are trainable at all.
5. **Gradient clipping** for explosions: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Standard practice in RNN and Transformer training.

**Real project example, recognising it in the wild:**

- Loss becomes `NaN` after a few iterations → exploding gradients. Lower the learning rate, add gradient clipping, check for `log(0)` in your loss.
- Loss decreases for a bit then completely flatlines while accuracy stays near chance → likely vanishing gradients. Check activations (are you using sigmoid in hidden layers?), check initialization, consider adding BatchNorm.

#### 2.1.17 Single neuron example (weight initialization intuition)

<a id="2117-single-neuron-example-weight-initialization-intuition"></a>

Take one neuron with n inputs and no bias:

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ
```

If each xᵢ has variance 1 and the wᵢ are independent, then `Var(z) = n · Var(w)`. So **the more inputs a neuron has, the larger its z becomes**: and large z means saturated activations and dead gradients.

To keep `Var(z) ≈ 1` regardless of n, you need `Var(w) = 1/n`. That's exactly Xavier initialization:

```text
W = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)
```

For ReLU, half the inputs get zeroed, halving the variance, so you compensate with a factor of 2. That's He initialization:

```text
W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
```

That's the entire derivation. It's not deep magic; it's variance bookkeeping. But it's the difference between a 30-layer network converging and not converging, which is why it's worth understanding rather than memorising.

### 2.2 Optimization Algorithms

<a id="22-optimization-algorithms"></a>

#### 2.2.1 Batch vs. mini-batch gradient descent

<a id="221-batch-vs-mini-batch-gradient-descent"></a>

**The idea in one line:** Instead of processing all 5 million examples before taking one step, process 512 at a time and take thousands of steps in the same wall-clock time.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Batch gradient descent** | Use the entire training set for each update. |
| **Mini-batch gradient descent** | Use a small subset (e.g. 64, 256) per update. |
| **Stochastic gradient descent (SGD)** | Mini-batch size of 1. (Confusingly, "SGD" in modern libraries usually means mini-batch.) |
| **Epoch** | One complete pass through the entire training set. |
| `X{t}` | The t-th mini-batch. Curly-brace superscript. |

**Deeper explanation:**

With m = 5,000,000, batch gradient descent requires the full forward and backward pass over all five million examples to move the parameters once. If that takes 15 minutes, you get 4 updates per hour. Convergence takes days.

Split into 5,000 mini-batches of 1,000. Now **one epoch gives you 5,000 parameter updates instead of 1.** The gradient from each mini-batch is a noisy estimate of the true gradient, but a noisy step in roughly the right direction, taken 5,000 times, beats one perfect step by an enormous margin.

**Notation:** the three superscript types now all appear together:

- `x⁽ⁱ⁾`, i-th example (round brackets)
- `z[l]`, l-th layer (square brackets)
- `X{t}`, t-th mini-batch (curly brackets)

#### 2.2.2 Mini-batch gradient descent, the algorithm

<a id="222-mini-batch-gradient-descent-the-algorithm"></a>

```text
for epoch in range(num_epochs):
    shuffle(X, Y)                        # reshuffle each epoch
    for t in range(num_minibatches):
        X_t, Y_t = get_minibatch(t)
        # forward prop on X_t
        # compute cost J{t} on this mini-batch
        # backprop to get gradients
        # update W, b
```

The inner loop body is identical to what you already know, only the data it sees per step is smaller.

**Shuffling each epoch matters.** Without it, the model sees mini-batches in the same order every epoch, which can introduce cyclic bias. Frameworks do this by default (`shuffle=True`); make sure it's on.

#### 2.2.3 Training with mini-batch gradient descent, reading the cost curve

<a id="223-training-with-mini-batch-gradient-descent-reading-the-cost-curve"></a>

**Batch gradient descent:** the cost decreases smoothly and monotonically every iteration. If it ever goes up, your learning rate is too high. Period.

**Mini-batch gradient descent:** the cost is **noisy**: it trends downward but jumps around. This is completely normal and expected. Some mini-batches happen to be harder than others, so `J{t}` naturally fluctuates.

**How to read the noisy curve in practice:** plot the cost per mini-batch and overlay a moving average. Judge convergence by the moving average, not the raw jitter. If the moving average has flattened, you're converged (or you need a lower learning rate to squeeze out more).

If the noise band is *very* wide, reduce the learning rate or increase the batch size, both reduce gradient variance.

#### 2.2.4 Choosing your mini-batch size

<a id="224-choosing-your-mini-batch-size"></a>

**The two extremes:**

| **Size** | **Name** | **Behaviour** |
| --- | --- | --- |
| m (all data) | Batch GD | Smooth, low noise, but each step is very slow. Only viable for small m. |
| 1 | Stochastic GD | Extremely noisy, never truly settles, and **loses all vectorization speedup**: you process one example at a time. |
| 64–512 | Mini-batch | The sweet spot: vectorization benefits plus fast, frequent updates. |

**Practical guidelines:**

1. **If m ≤ 2,000, just use batch gradient descent.** The whole dataset fits comfortably; mini-batching adds complexity for nothing.
2. **Otherwise use powers of 2: 64, 128, 256, 512.** Powers of two align with memory layout and GPU architecture, giving genuinely better throughput.
3. **Make sure a mini-batch fits in GPU memory.** If it doesn't, performance collapses catastrophically as data thrashes between CPU and GPU. This is the single most common practical constraint on batch size.
4. **Treat it as a hyperparameter** and try a few values.

**Rules of thumb from practice:** larger batches give more stable gradients and better GPU utilisation but sometimes generalise slightly worse. Smaller batches add gradient noise, which can act as a mild regulariser. A useful heuristic: when you increase the batch size by k, consider increasing the learning rate by roughly k (or √k) to compensate for the reduced gradient noise.

**Real project example:** Training an image classifier on a single 16GB GPU with 224×224 images and a ResNet-50, batch size 32 is typical. Push to 64 and you'll hit an out-of-memory error. If you need a larger effective batch, use **gradient accumulation**: run 4 forward/backward passes of 32, accumulate the gradients, then update once. Effective batch of 128 with the memory footprint of 32.

#### 2.2.5 Exponentially weighted averages

<a id="225-exponentially-weighted-averages"></a>

**The idea in one line:** A running average that remembers the past with exponentially decaying weight, the building block for every advanced optimizer.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Exponentially weighted (moving) average** | A smoothed version of a noisy sequence. |
| **β (beta)** | How much history to keep. Higher = smoother, more lag. |

**The formula:**

```text
v_t = β·v_{t−1} + (1−β)·θ_t
```

where `θ_t` is the new raw observation and `v_t` is the smoothed value.

**Example, daily temperature in London.** Raw daily temperatures are noisy: 4°, 9°, 7°, 12°, 6°...

```text
v₀ = 0
v₁ = 0.9·v₀ + 0.1·θ₁
v₂ = 0.9·v₁ + 0.1·θ₂
...
```

With β = 0.9 you get a smooth curve tracking the seasonal trend and ignoring day-to-day noise.

**The rule of thumb:** `v_t` **averages roughly the last** `1/(1−β)` **observations.**

| **β** | **Approx. window** | **Behaviour** |
| --- | --- | --- |
| 0.5 | 2 days | Very noisy, tracks changes instantly |
| 0.9 | 10 days | Good balance, the standard default |
| 0.98 | 50 days | Very smooth, but lags noticeably behind real shifts |

The lag is the tradeoff. High β adapts slowly: when the real temperature drops sharply, a β=0.98 curve takes many days to catch up.

#### 2.2.6 Understanding exponentially weighted averages

<a id="226-understanding-exponentially-weighted-averages"></a>

**Why "exponentially weighted"?** Expand the recursion:

v₁₀₀ = 0.1·θ₁₀₀ + 0.1·0.9·θ₉₉ + 0.1·0.9²·θ₉₈ + 0.1·0.9³·θ₉₇ +...

Each older observation is weighted by an additional factor of 0.9. The weights decay **exponentially** into the past. Recent data matters most; ancient data fades to irrelevance.

**Where does** `1/(1−β)` **come from?** Because `0.9^10 ≈ 0.35 ≈ 1/e`. After about 10 steps the weight

has decayed to roughly a third, so contributions beyond that are small. In general `β^(1/(1−β)) ≈ 1/e`,

which is why the window is about `1/(1−β)`.

#### 2.2.7 Implementing exponentially weighted averages

<a id="227-implementing-exponentially-weighted-averages"></a>

```text
v = 0
for t in range(1, T+1):
    v = beta * v + (1 - beta) * theta[t]
```

**The key practical virtue: one line, one variable, O(1) memory.** A true moving average over 10 days requires storing 10 values and re-summing. This requires one float. When you're tracking a running average for *every parameter in a 100-million-parameter model*, that memory difference is the whole ballgame. It's slightly less accurate than a true average, and completely worth it.

#### 2.2.8 Bias correction

<a id="228-bias-correction"></a>

**The problem:**`v₀` `= 0`, so early estimates are biased toward zero.

With β = 0.98 and θ₁ = 40:

```text
v₁ = 0.98(0) + 0.02(40) = 0.8      ← should be ≈ 40, not 0.8!
```

The estimate is far too low for the first several dozen steps.

**The fix, divide by** `(1 − β`**ᵗ**`)`**:**

```text
v_corrected_t = v_t / (1 − βᵗ)
```

Check it: at t=1, `1 − 0.98¹ = 0.02`, so `0.8 / 0.02 = 40`. Correct.

As t grows, `βᵗ→ 0` and the denominator → 1, so the correction fades away automatically. It matters early and disappears later, exactly the behaviour you want.

**In practice:** many people skip bias correction and simply accept that the first few dozen iterations are off. But **Adam includes it by default**, and it does help early training stability, so there's no reason to omit it.

#### 2.2.9 Gradient descent with momentum

<a id="229-gradient-descent-with-momentum"></a>

**The idea in one line:** Instead of stepping in the direction of the current gradient, step in the direction of a smoothed average of recent gradients, which cancels oscillation and accelerates consistent progress.

**The algorithm:**

```text
v_dW = β·v_dW + (1−β)·dW
v_db = β·v_db + (1−β)·db
W := W − α·v_dW
b := b − α·v_db
```

**Why it helps:**

Picture an elongated cost surface (an ellipse). Plain gradient descent oscillates up and down across the narrow axis while making slow progress along the long axis. Two things happen with momentum:

- **In the oscillating direction**, consecutive gradients point up, down, up, down. Averaging them mostly **cancels them out**. The oscillation damps.
- **In the progress direction**, consecutive gradients all point the same way. Averaging **reinforces** them. You accelerate.

The physical analogy is a ball rolling down a bowl. `dW` is acceleration, `v_dW` is velocity, and β acts like friction. The ball doesn't jitter side to side; it builds speed downhill.

The net effect is that you can use a larger learning rate without divergence, so you converge faster.

**Hyperparameter:** β = 0.9 is a robust default, essentially never needs tuning.

**Note on a common variant:** some formulations drop the `(1−β)` factor, writing `v_dW = β·v_dW + dW`. It

works too, but rescales the effective learning rate by `1/(1−β)`, so α must be retuned if you switch conventions. Be aware which one your framework uses.

#### 2.2.10 RMSprop

<a id="2210-rmsprop"></a>

*(The syllabus lists this as "PMSprop", that's a typo. It's RMSprop: Root Mean Square propagation.)*

**The idea in one line:** Give each parameter its own adaptive learning rate by dividing by the recent magnitude of that parameter's gradients.

**The algorithm:**

```text
s_dW = β₂·s_dW + (1−β₂)·(dW)²        ← elementwise square
s_db = β₂·s_db + (1−β₂)·(db)²
W := W − α · dW / (√s_dW + ε)
b := b − α · db / (√s_db + ε)
```

**Why it helps:**

Suppose parameter `b` oscillates violently (large gradients) while `W` moves slowly (small gradients). Then

`s_db` is large and `s_dW` is small. Dividing by `√s`:

- `b`'s effective step gets **smaller** → oscillation damped.
- `W`'s effective step gets **larger** → progress accelerated.

So the optimizer automatically slows down in steep directions and speeds up in flat ones. It normalises the step size per parameter.

**The ε:** typically 1e-8. Purely to prevent division by zero when a gradient has been near zero for a while. It has no conceptual role.

**Momentum vs. RMSprop, in one line each:** Momentum smooths the *direction* of the step. RMSprop adapts the *size* of the step per parameter. They address different problems, which is why combining them is such a good idea.

#### 2.2.11 Adam optimization

<a id="2211-adam-optimization"></a>

**The idea in one line:** Momentum + RMSprop + bias correction, combined. The default optimizer for essentially all of deep learning.

**Jargon decoded:Adam** = **Ada**ptive **M**oment estimation. "Moments" is statistics jargon: the first moment is the mean (that's the momentum term), the second moment is the uncentred variance (that's the RMSprop term).

**The full algorithm:**

```text
Initialize: v_dW = 0, s_dW = 0, v_db = 0, s_db = 0
On iteration t:
    compute dW, db on the current mini-batch
    # Momentum (first moment)
    v_dW = β₁·v_dW + (1−β₁)·dW
    v_db = β₁·v_db + (1−β₁)·db
    # RMSprop (second moment)
    s_dW = β₂·s_dW + (1−β₂)·(dW)²
    s_db = β₂·s_db + (1−β₂)·(db)²
    # Bias correction
    v_dW_corr = v_dW / (1 − β₁ᵗ)
    v_db_corr = v_db / (1 − β₁ᵗ)
    s_dW_corr = s_dW / (1 − β₂ᵗ)
    s_db_corr = s_db / (1 − β₂ᵗ)
    # Update
    W := W − α · v_dW_corr / (√s_dW_corr + ε)
    b := b − α · v_db_corr / (√s_db_corr + ε)
```

**Hyperparameter choices (the "traditionally used" values from the syllabus):**

| **Hyperparameter** | **Value** | **Tune it?** |
| --- | --- | --- |
| α (learning rate) | problem-specific | **Yes, always tune this** |
| β₁ | 0.9 | No, default is fine |
| β₂ | 0.999 | No, default is fine |
| ε | 10⁻⁸ | No, never matters |

**This is the practical takeaway of the entire optimization section:** with Adam, you tune exactly one optimizer hyperparameter, the learning rate. The other three have defaults that work across an extraordinary range of problems, which is precisely why Adam became universal.

**Real project example:**`optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)` and

`keras.optimizers.Adam(learning_rate=0.001)`. Adam with lr=0.001 is a genuinely good starting point for almost anything. If training is unstable, drop to 1e-4. For fine-tuning a pretrained model, use something much smaller, 1e-5, because you want to nudge, not overwrite, what the model already knows.

**Worth knowing:AdamW** decouples weight decay from the adaptive scaling and generally generalises better. It's the default for Transformer training and increasingly for everything else. If your framework offers it, prefer it.

#### 2.2.12 Learning rate decay

<a id="2212-learning-rate-decay"></a>

**The idea in one line:** Start with big steps to cover ground fast, then shrink the steps so you can settle precisely into the minimum.

**Deeper explanation:**

With a fixed learning rate and mini-batches, the gradient noise never goes away, so near the minimum you don't converge, you wander around in a loose orbit whose radius is proportional to α. Shrinking α over time tightens that orbit and lets you actually settle.

**Common schedules:**

```python
# 1. Inverse time decay
alpha = alpha0 / (1 + decay_rate * epoch)
# 2. Exponential decay
alpha = (0.95 ** epoch) * alpha0
# 3. Inverse square root
alpha = alpha0 * k / np.sqrt(epoch)
# 4. Step / staircase decay - halve α every N epochs
# 5. Manual decay - watch the curve and drop α by hand when it plateaus
```

**Also standard in modern practice:**

- **Cosine annealing**: smoothly decay α along a cosine curve to near zero. Very common and very effective.
- **Warmup**: start α *very* small for the first few hundred steps and ramp it up, then decay. Essential for Transformer training, where early large steps destabilise everything.
- **ReduceLROnPlateau**: automatically cut α by a factor (e.g. 0.5) whenever dev loss stops improving for N epochs. The most "set and forget" option.

**Where it ranks in priority:** below the base learning rate, and below architecture. Get α right first; add a schedule to squeeze out the last few percent.

**Real project example:**`keras.callbacks.ReduceLROnPlateau(monitor='val_loss',`

`factor=0.5, patience=5)`. Pair it with EarlyStopping and you have a training loop that manages its own learning rate and knows when to quit, two lines, and it reliably beats a fixed learning rate.

#### 2.2.13 The problem of local optima

<a id="2213-the-problem-of-local-optima"></a>

**The idea in one line:** The classic fear of getting trapped in a local minimum is largely a myth in high dimensions, the real enemy is plateaus.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Local optimum** | A point that's lowest in its immediate neighbourhood but not globally lowest. |
| **Saddle point** | A point where the gradient is zero, but it curves *up* in some directions and *down* in others, like the middle of a horse saddle. |
| **Plateau** | A large flat region where gradients are tiny and progress is glacial. |

**Deeper explanation:**

The intuition people carry comes from 2D pictures: a wiggly curve with several valleys, and gradient descent falling into the wrong one. That picture is misleading, because your parameter space has millions of dimensions, not two.

For a point to be a true local minimum in 20,000 dimensions, the function must curve **upward in all 20,000 directions simultaneously.** If each direction were an independent coin flip, that's roughly 2^(−20,000), effectively impossible.

What you almost always find instead is a **saddle point**: gradient zero, curving up in some directions and down in others. And saddle points are escapable, there's always a downhill direction available.

**So the real problem is plateaus.** Long flat stretches where the gradient is nearly zero in every direction. Learning doesn't stop, but it crawls, and you may sit there for thousands of iterations before the surface tilts again.

**This is precisely what momentum, RMSprop, and Adam are for.** Momentum carries you across flat regions using accumulated velocity. RMSprop *amplifies* small gradients (dividing by a small `√s`), which is exactly the behaviour you need on a plateau. So the optimizers of section 2.2 are not just speedups, they're the direct answer to the geometry of high-dimensional loss surfaces.

**Real project example:** If your loss sits flat for 500 iterations and you're using plain SGD, switch to Adam before concluding your architecture is wrong. Then check initialization. Then check that your data is actually normalised. Only then question the model. A surprising fraction of "the model won't learn" reports are one of these three.

### 2.3 Hyperparameter Tuning, Batch Normalization & Multi-Class Classification

<a id="23-hyperparameter-tuning-batch-normalization-multi-class-classification"></a>

#### 2.3.1 The hyperparameter tuning process

<a id="231-the-hyperparameter-tuning-process"></a>

**The idea in one line:** Not all hyperparameters matter equally, tune the important ones first, and don't waste search budget on the rest.

**The priority ordering (roughly, from most to least important):**

**Tier 1, always tune:**

- Learning rate α

**Tier 2, usually worth tuning:**

- Mini-batch size
- Number of hidden units
- Momentum β (if not using Adam)

**Tier 3, sometimes:**

- Number of layers
- Learning rate decay schedule

**Tier 4, almost never:**

- Adam's β₁ (0.9), β₂ (0.999), ε (10⁻⁸)

**The practical implication:** if you have budget for 20 experiments, spend 10 on the learning rate. Spending them uniformly across all hyperparameters is a classic beginner mistake that wastes most of the budget on parameters whose defaults were already fine.

#### 2.3.2 Random values, not a grid

<a id="232-random-values-not-a-grid"></a>

**The idea in one line:** Sample hyperparameters randomly rather than on a regular grid, you'll explore far more distinct values of the parameters that actually matter.

**Deeper explanation:**

Say you're tuning two hyperparameters with 25 experiments.

**Grid search (5×5):** you try 5 distinct values of hyperparameter 1 and 5 of hyperparameter 2.

**Random search (25 random points):** you try **25 distinct values of each**.

Now here's the point. Suppose hyperparameter 1 is the learning rate (hugely important) and hyperparameter 2 is ε (irrelevant). Grid search wasted 20 of its 25 runs re-testing the same 5 learning rates with different useless ε values. **Random search tested 25 different learning rates.** Five times the effective resolution on the parameter that mattered, and you didn't have to know in advance which one that was.

Since you generally *don't* know in advance which hyperparameters matter for a new problem, random search dominates. This is one of the most well-supported empirical results in ML practice.

**Coarse-to-fine refinement:** run a broad random search, find the region where results cluster well, then zoom in and sample densely inside that smaller region. Repeat.

**Real project example:**`sklearn.model_selection.RandomizedSearchCV`, or better for deep learning, **Optuna** or **Ray Tune**, which add Bayesian optimisation, using earlier results to decide where to sample next. Optuna in particular is a small amount of code for a large improvement over manual search.

#### 2.3.3 Using an appropriate scale to pick hyperparameters

<a id="233-using-an-appropriate-scale-to-pick-hyperparameters"></a>

**The idea in one line:** Sample on a log scale for anything that spans orders of magnitude, otherwise you'll spend almost all your samples in the wrong region.

**Deeper explanation:**

**When a linear scale is right:** number of hidden units (50–100), number of layers (2–4). Uniform sampling is fine.

**When a linear scale is wrong: the learning rate.** Suppose you sample uniformly between 0.0001 and 1. Then 90% of your samples land between 0.1 and 1, and only 0.01% land between 0.0001 and 0.001, which is often where the answer lives. You've wasted nearly the entire budget.

**The fix, sample uniformly in log space:**

```python
r = -4 * np.random.rand()      # r uniform in [-4, 0]
alpha = 10 ** r                # alpha in [1e-4, 1]
```

Now you get equal numbers of samples in [0.0001, 0.001], [0.001, 0.01], [0.01, 0.1], and [0.1, 1]. Every order of magnitude is explored equally.

#### 2.3.4 Hyperparameters for exponentially weighted averages

<a id="234-hyperparameters-for-exponentially-weighted-averages"></a>

**The special case of β:** you want to search β between 0.9 and 0.999.

Don't sample β directly. Sample `1 − β` on a log scale:

```python
r = -3 * np.random.rand() - 1      # r uniform in [-4, -1]
one_minus_beta = 10 ** r           # in [1e-4, 1e-1]
beta = 1 - one_minus_beta          # in [0.9, 0.9999]
```

**Why this matters so much:** recall that the averaging window is `1/(1−β)`.

- β = 0.9000 → 10-step window
- β = 0.9005 → 10.05-step window, *essentially no change*
- β = 0.9990 → 1000-step window
- β = 0.9995 → 2000-step window, ***the window doubled***

A change of 0.0005 is meaningless near 0.9 and enormous near 0.999. Linear sampling in β treats those two regions identically, which is exactly wrong. Sampling `1−β` logarithmically gives you resolution proportional to actual effect.

**The general principle:** sample on the scale where equal steps produce equal changes in behaviour, not on the scale that looks natural.

#### 2.3.5 Re-test hyperparameters occasionally

<a id="235-re-test-hyperparameters-occasionally"></a>

**The idea in one line:** Hyperparameters go stale. Data drifts, hardware changes, code changes. Retune every few months.

**The two organisational styles:**

**Panda (babysitting one model):** You have limited compute. Train one model and watch it daily, nudging the learning rate up or down, reverting if it goes badly. Slow but resource-efficient. Named for pandas raising a single cub with great care.

**Caviar (training many in parallel):** You have plenty of compute. Launch 50 models with different hyperparameters simultaneously and pick the winner. Named for fish laying millions of eggs and letting the best survive.

Choose based on your compute budget relative to model size. Most startups are pandas by necessity; a research lab training small models is a caviar shop.

**Real project example:** A production churn model trained on 2024 data will degrade in 2025 as your product, pricing, and customer mix change. Schedule quarterly retraining *and* a hyperparameter re-search. Also monitor for **data drift** in production, track the distribution of input features over time and alert when they shift meaningfully. A model whose inputs have drifted is a model whose accuracy is quietly falling while your dashboard still shows the number from launch day.

#### 2.3.6 Batch normalization, normalizing activations in a network

<a id="236-batch-normalization-normalizing-activations-in-a-network"></a>

**The idea in one line:** We normalise the *inputs* to the network to speed up training, batch norm does the same thing to the inputs of *every hidden layer*.

**Deeper explanation:**

Section 2.1.15 established that normalising input features makes the cost surface round and training fast. Batch norm asks the obvious follow-up: why stop at layer 0? Layer 3's inputs are layer 2's activations, and those can be badly scaled too.

**What gets normalised:** the standard is to normalise `z[l]` (the pre-activation), not `a[l]`. This is the more common choice in practice and in the original paper.

#### 2.3.7 Implementing batch norm

<a id="237-implementing-batch-norm"></a>

Given the values `z⁽¹⁾,..., z⁽ᵐ⁾` for some layer across the current mini-batch:

μ = (1/m)·Σ z⁽ⁱ⁾ # mean over the mini-batch σ² = (1/m)·Σ (z⁽ⁱ⁾ − μ)² # variance over the mini-batch z_norm⁽ⁱ⁾ = (z⁽ⁱ⁾ − μ) / √(σ² + ε) # now mean 0, variance 1 z̃⁽ⁱ⁾ = γ · z_norm⁽ⁱ⁾ + β # rescale and re-shift

**The crucial part is the last line, and it's where beginners get confused.**

After normalising, every layer's inputs have mean 0 and variance 1. But that's not always what you want! With a sigmoid activation, mean 0 / variance 1 keeps you pinned in the *linear middle* of the sigmoid, wasting its non-linearity entirely.

So batch norm introduces **γ (gamma)** and **β (beta)**: two **learnable parameters, one pair per layer**: that let the network choose its own mean and variance for that layer. If mean 0 / variance 1 is best, the network learns γ = 1, β = 0. If a different distribution is better, it learns that instead.

The point isn't to force a specific distribution. **The point is to make the distribution a directly learnable parameter instead of an unstable emergent side-effect of every other weight in the network.**

**Naming collision warning:** this `β` is *not* the momentum/Adam β. Completely different thing, same Greek letter. Blame convention.

#### 2.3.8 Fitting batch norm into a neural network

<a id="238-fitting-batch-norm-into-a-neural-network"></a>

The layer computation changes from:

```text
z[l] = W[l]a[l−1] + b[l]   →   a[l] = g(z[l])
```

to:

```text
z[l] = W[l]a[l−1]          →   z̃[l] = BatchNorm_{γ[l],β[l]}(z[l])   →   a[l] = g(z̃[l])
```

**Notice** `b[l]` **disappeared.** This is not a mistake. Batch norm subtracts the mean of z across the

mini-batch, and subtracting the mean removes any constant offset. So `b[l]` gets cancelled out entirely; it

has no effect. Its role is taken over by `β[l]`, which does the same job but is applied after normalisation where it actually survives.

**So when using batch norm, drop the bias term.** In Keras that's `Dense(64, use_bias=False)`

followed by `BatchNormalization()`. It saves parameters and avoids a redundant, useless variable.

The learnable parameters of the network are now `W[l], γ[l], β[l]` for each layer, all updated by gradient descent (or Adam) exactly like the weights.

#### 2.3.9 Working with mini-batches

<a id="239-working-with-mini-batches"></a>

Batch norm computes μ and σ² **over the current mini-batch only**: hence the name.

This has a real consequence: **the same input produces different activations depending on which other examples share its mini-batch.** That's the source of both batch norm's regularization effect (below) and its main weakness.

**The weakness in practice:** with very small batch sizes (1–4), the mini-batch statistics are extremely noisy estimates and batch norm degrades badly. This is common when training large models where memory forces tiny batches. The fix is to use a batch-size-independent alternative:

- **Layer Normalization**: normalise across features within a single example. Batch-size independent. Standard in Transformers and RNNs.
- **Group Normalization**: normalise across groups of channels. Popular in vision with small batches.

#### 2.3.10 Implementing gradient descent with batch norm

<a id="2310-implementing-gradient-descent-with-batch-norm"></a>

```text
for each mini-batch t:
    forward prop on X{t}, applying batch norm at each layer
    backprop to compute dW[l], dγ[l], dβ[l]
    update:  W[l] := W[l] − α·dW[l]
             γ[l] := γ[l] − α·dγ[l]
             β[l] := β[l] − α·dβ[l]
```

γ and β are ordinary parameters. They're learned the same way as weights, and work with momentum, RMSprop, and Adam without modification. Frameworks handle all of this, `BatchNormalization()` is one line, but knowing that γ and β are *learned* rather than fixed is what makes the layer make sense.

#### 2.3.11 Why does batch norm work?

<a id="2311-why-does-batch-norm-work"></a>

**Reason 1, the same reason input normalization works.** Every layer now receives well-scaled inputs, so its own optimisation problem is better conditioned. You can use a larger learning rate and converge faster. In practice, batch norm often lets you increase α by an order of magnitude.

**Reason 2, it reduces internal covariate shift.**

**Jargon decoded:Covariate shift** means the input distribution changes while the underlying input→output relationship stays the same. Classic example: you train a cat classifier on only black cats, then test it on ginger cats. The mapping "this shape is a cat" hasn't changed, but the input distribution has, and your model breaks.

**Why this is a problem *inside* a neural network:** layer 4 is trying to learn a mapping from its inputs (layer 3's activations) to its outputs. But layer 3's weights are *also being updated every iteration*, so the distribution of layer 4's inputs keeps shifting under its feet. Layer 4 is chasing a moving target.

Batch norm doesn't stop layer 3's values from changing, but it **guarantees their mean and variance stay stable** (at whatever γ and β specify). Layer 4's inputs now shift much less, so layer 4's learning problem is more stable. Layers become more independent of each other, which speeds up learning across the whole network.

*(Honest caveat: subsequent research has questioned whether covariate shift is really the mechanism, suggesting instead that batch norm simply smooths the loss landscape. The empirical benefit is not in doubt; the explanation is still debated. Worth knowing so you're not surprised when you read conflicting accounts.)*

#### 2.3.12 Batch norm as regularization

<a id="2312-batch-norm-as-regularization"></a>

A side effect, and a real one:

- Each mini-batch is scaled by the mean and variance computed **on just that mini-batch**, which are noisy estimates of the true statistics.
- This adds noise to `z[l]` for every example, similar in spirit to dropout, which adds noise by zeroing units.
- Noise forces the downstream layers not to over-rely on any single activation value.
- Therefore: **a slight regularization effect.**

**Two important qualifications:**

1. **The effect is small**, and it shrinks as batch size grows (bigger batches → less noisy statistics → less regularization). Don't use batch norm *as* your regularizer; use it for optimization speed and treat the regularization as a bonus.
2. **Don't rely on it.** Use dropout and L2 for actual regularization. That said, batch norm and dropout together can interact awkwardly, and many modern architectures use batch norm alone, or batch norm plus weight decay. If you're using both and results are odd, try removing dropout.

#### 2.3.13 Batch norm at test time

<a id="2313-batch-norm-at-test-time"></a>

**The problem:** at test time you may be predicting for a **single example**. What is "the mean and variance of a mini-batch of one"? The mean is the value itself, and the variance is zero. Normalising by that is meaningless.

**The solution:** during training, keep a running exponentially weighted average of μ and σ² across all mini-batches. At test time, use those saved values instead of computing fresh ones.

```text
# During training, for each layer l:
running_mu  = β_ema · running_mu  + (1 − β_ema) · μ{t}
running_var = β_ema · running_var + (1 − β_ema) · σ²{t}
# At test time:
z_norm = (z − running_mu) / √(running_var + ε)
z̃      = γ · z_norm + β
```

These running statistics are **not learned by gradient descent**: they're accumulated statistics, stored as part of the model. (In PyTorch they show up as "buffers" rather than "parameters," which is exactly this distinction.)

**Real project example, the classic production bug:** forgetting `model.eval()` in PyTorch means batch norm keeps using batch statistics at inference. If you're serving one request at a time, that means normalising by the statistics of a single example, and your production accuracy collapses versus your validation accuracy, with no error message. Same call also disables dropout. `model.eval()` **before**

**inference,** `model.train()` **before training.** Write it into your serving code and never think about it again.

#### 2.3.14 Multi-class classification and softmax regression

<a id="2314-multi-class-classification-and-softmax-regression"></a>

**The idea in one line:** When there are more than two classes, the output layer produces one probability per class, all summing to 1.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Multi-class classification** | Pick exactly one of C classes. |
| **C** | Number of classes. |
| **Softmax** | The activation that turns C raw scores into C probabilities summing to 1. |
| **Logits** | The raw pre-softmax scores `z[L]`. Any real numbers. |
| **One-hot encoding** | Representing class 2 of 4 as [0, 1, 0, 0]. |
| **Hard-max** | The (non-softmax) alternative: put 1 on the largest, 0 elsewhere. Not differentiable, so not used for training. |

**Deeper explanation:**

The output layer has C units instead of 1, so `z[L]` has shape (C, 1). Then:

```text
t = e^(z[L])                          # elementwise exponential, shape (C,1)
a[L] = t / Σⱼ tⱼ                      # divide by the sum → shape (C,1)
```

**Worked example.** Say `z[L] = [5, 2, −1, 3]`:

```text
t    = [e⁵, e², e⁻¹, e³] = [148.4, 7.4, 0.37, 20.1]
sum  = 176.3
a[L] = [0.842, 0.042, 0.002, 0.114]
```

Sums to 1. Largest logit gets the largest probability. Done.

**Why the exponential?** Two jobs. It makes everything positive (probabilities can't be negative, and logits can), and it **amplifies differences**: a logit that's larger by 3 becomes e³ ≈ 20× more probable. This makes the model decisive rather than mushy.

**Softmax is unusual among activations** in one respect: sigmoid, ReLU, and tanh operate on each element independently. Softmax needs the *whole vector*, because of the normalising sum. Raising one logit lowers every other probability. That's intentional, the classes compete, which is exactly right when only one can be correct.

#### 2.3.15 The softmax layer and classifier

<a id="2315-the-softmax-layer-and-classifier"></a>

**The decision boundaries:** with **no hidden layers**, softmax regression produces *linear* decision boundaries between every pair of classes, it's the direct generalisation of logistic regression to C classes. With hidden layers, the boundaries become arbitrarily complex.

**Sanity check: when C = 2, softmax reduces exactly to logistic regression.** Two outputs summing to 1 carry the same information as one output in [0,1]. The extra parameter is redundant. So logistic regression really is the special case, not a different algorithm.

#### 2.3.16 The softmax loss function

<a id="2316-the-softmax-loss-function"></a>

**For a single example:**

L(ŷ, y) = − Σⱼ yⱼ · log(ŷⱼ)

Since `y` is one-hot, all terms vanish except the one for the true class. So it simplifies to:

```text
L = − log(ŷ_correct_class)
```

**The whole loss function reduces to: make the probability of the correct class as high as possible.** Nothing else. It doesn't care how the remaining probability is distributed among wrong classes, and it doesn't need to, since they all sum to 1 anyway.

This is called **categorical cross-entropy**.

**And the gradient is, once again:**

```text
dz[L] = ŷ − y
```

The same clean form as binary logistic regression, and for the same underlying reason: the softmax activation and cross-entropy loss are mathematically matched, and all the messy derivative terms cancel.

**Real project example (CatShop, C = 47 product categories):**

- Output layer: `Dense(47, activation='softmax')`
- Loss: `categorical_crossentropy` if labels are one-hot, or `sparse_categorical_crossentropy` if they're plain integers (0–46). The sparse version saves you from building 47-wide one-hot vectors, use it.
- **Numerical stability note:** frameworks provide a `from_logits=True` option that lets you skip the softmax activation and pass raw logits to the loss, which computes softmax+log together in a numerically stable way. Prefer it. `e^1000` overflows to infinity; the fused version avoids that entirely.

**Multi-label is a different problem.** If a product photo can be *both* "outdoor" and "sports equipment," those classes don't compete, use **C independent sigmoid outputs** with `binary_crossentropy`, not softmax. Softmax forces the probabilities to sum to 1, which is wrong when multiple labels can be simultaneously true. This is a genuinely common modelling error.

### Deep Learning Frameworks

<a id="deep-learning-frameworks"></a>

#### Choosing a framework

<a id="choosing-a-framework"></a>

**The three criteria from the syllabus:**

1. **Ease of programming**: for both development *and deployment*. Deployment is easy to underweight and expensive to get wrong.
2. **Running speed**: training throughput and inference latency.
3. **Truly open**: open source *with good governance*. Open source alone isn't enough; a company can open-source something and then close it later. Look at governance, not just the licence.

**The 2026 landscape** (the syllabus list, Caffe, CNTK, DL4J, Keras, Lasagne, mxnet, PaddlePaddle, TensorFlow, Theano, Torch, reflects an older era; several are now discontinued):

| **Framework** | **Where it stands** |
| --- | --- |
| **PyTorch** | Dominant in research and increasingly in production. Pythonic, easy to debug. **Default recommendation for learning.** |
| **TensorFlow / Keras** | Strong in production and mobile/edge deployment. Keras is the friendliest high-level API anywhere. |
| **JAX** | Functional, extremely fast, popular for large-scale research. Steeper curve. |
| **Hugging Face Transformers** | Not a framework but the standard layer above them for pretrained models. Essential in practice. |

**Practical advice:** learn PyTorch first. The concepts transfer entirely, every framework implements the same forward/backward/optimizer loop you now understand from Chapter 1.

#### The TensorFlow code example, explained

<a id="the-tensorflow-code-example-explained"></a>

The syllabus contains this code, which is **TensorFlow 1.x**: deprecated since 2019, but worth reading because it makes the computation graph explicit:

```python
import numpy as np
import tensorflow as tf
coefficients = np.array([[1], [-20], [25]])
w = tf.Variable([0], dtype=tf.float32)
x = tf.placeholder(tf.float32, [3,1])
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]      # (w-5)**2
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        session.run(train, feed_dict={x: coefficients})
    print(session.run(w))
```

**What it does:** minimises `w² − 20w + 25 = (w−5)²`, whose minimum is at w = 5. After 1000 steps it prints approximately 4.9999.

**What matters conceptually:**

1. **You only specified the forward computation** (`cost`). TensorFlow derived the backward pass automatically. That's the payoff of the computation-graph view from section 1.1.9, you define the graph, the framework handles differentiation.
2. `placeholder` **+** `feed_dict` is how TF1 got data in, you built a static graph first, then fed values through it. Clunky, and the main reason TF1 was hard to debug.
3. `Session` existed because the graph was defined first and executed later ("define-then-run"). You couldn't just print an intermediate value.

**The same thing in modern TensorFlow 2:**

```python
import tensorflow as tf
w = tf.Variable(0.0)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
coefficients = tf.constant([1.0, -20.0, 25.0])
for i in range(1000):
    with tf.GradientTape() as tape:
        cost = coefficients[0]*w**2 + coefficients[1]*w + coefficients[2]
    grads = tape.gradient(cost, [w])
    optimizer.apply_gradients(zip(grads, [w]))
print(w.numpy())     # ≈ 5.0
```

**And in PyTorch:**

```python
import torch
w = torch.tensor([0.0], requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.01)
for i in range(1000):
    optimizer.zero_grad()
    cost = w**2 - 20*w + 25
    cost.backward()          # ← autograd computes dcost/dw
    optimizer.step()
print(w.item())     # ≈ 5.0
```

Notice how directly the PyTorch version maps onto the gradient descent loop you learned in section 1.1.8: zero the gradients, compute the cost, backprop, step. That correspondence is why understanding the fundamentals makes frameworks feel obvious rather than magical.

**One PyTorch gotcha:**`optimizer.zero_grad()` is mandatory. PyTorch *accumulates* gradients by default (which is what enables gradient accumulation for large effective batch sizes). Forget it, and your gradients pile up across iterations and training diverges. It's the most common PyTorch beginner bug.

## Chapter 4: Structuring Machine Learning Projects

<a id="chapter-4-structuring-machine-learning-projects"></a>

*This chapter contains no new math. It is entirely about decision-making, and in real projects it is often worth more than everything in Chapters 1 and 2 combined, because it determines whether your effort goes somewhere useful.*

### 4.1 Introduction to ML Strategy

<a id="41-introduction-to-ml-strategy"></a>

#### 4.1.1 Why ML strategy?

<a id="411-why-ml-strategy"></a>

**The idea in one line:** When your model is stuck, you have twenty plausible next moves and only one is right, strategy is how you pick without burning three months.

**Deeper explanation:**

Your cat classifier sits at 90% accuracy. Your options:

- Collect more data
- Collect a more diverse training set
- Train longer with gradient descent
- Try Adam instead of plain gradient descent
- Try a bigger network
- Try a smaller network
- Try dropout
- Add L2 regularization
- Change the architecture (activation functions, number of hidden units,...)

Any of these could be the answer. Most of them aren't. **Six months spent collecting more data when your problem was bias is six months gone**, and this happens to real teams constantly.

The goal of ML strategy is fast, well-reasoned ways to decide which ideas to pursue and which to discard.

#### 4.1.2 Orthogonalization

<a id="412-orthogonalization"></a>

**The idea in one line:** Each knob should control exactly one thing, so you can fix one problem without breaking another.

**Jargon decoded:Orthogonal** means "at right angles", in this context, independent. Adjusting one thing doesn't disturb another.

**Deeper explanation:**

The analogy is an old TV set with separate knobs for horizontal position, vertical position, width, height, and rotation. Each knob does one job. If instead you had one knob that adjusted "0.3 × height + 0.7 × rotation," tuning the picture would be nearly impossible, every adjustment would break something else.

Or a car: steering, accelerator, brake. Three controls, three effects. Imagine a car where one joystick controlled `0.3 × steering + 0.8 × speed`. Technically you could still drive it. Practically, you'd crash.

**The chain of assumptions in ML, and its dedicated knobs:**

| **Assumption** | **If it fails, turn this knob** |
| --- | --- |
| 1. Fit the **training set** well on the cost function | Bigger network, better optimizer (Adam), train longer |
| 2. Fit the **dev set** well | Regularization, bigger training set |
| 3. Fit the **test set** well | Bigger dev set (you overfit the dev set by tuning too much) |
| 4. Perform well **in the real world** | Change the dev set, or change the cost function |

Each row has its own tools. **Diagnose which row is broken, then use only that row's tools.**

**Why early stopping is criticised here:** it is explicitly *less* orthogonal. Stopping early simultaneously worsens the training fit (row 1) and improves the dev fit (row 2). One knob, two effects, you can't reason cleanly about it. That's the theoretical objection. In practice most teams use it anyway because it's cheap and effective, but the criticism is legitimate and worth understanding.

**Real project example (SaaSChurn):** Dev AUC is disappointing at 0.74. Before touching anything, check train AUC.

- Train 0.76 → row 1 is broken. Bigger network. **Do not add dropout**: that would make it worse.
- Train 0.95 → row 2 is broken. Regularization or more data. **Do not add layers**: that would make it worse.

The exact same symptom demands opposite treatments. That's the entire value of orthogonalization: it stops you from applying the right fix to the wrong problem.

### Setting Up Your Goal

<a id="setting-up-your-goal"></a>

#### 4.1.3 Single number evaluation metric

<a id="413-single-number-evaluation-metric"></a>

**The idea in one line:** Reduce model quality to one number, because comparing on two numbers is slow and often impossible.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Precisio n** | Of the examples you *called* positive, what fraction actually were? "When I say cat, how often am I right?" |
| **Recall** | Of the examples that *actually* were positive, what fraction did you catch? "Of all real cats, how many did I find?" |
| **F1 score** | The harmonic mean of precision and recall: `2 / (1/P + 1/R)`. A single number combining both. |

**Deeper explanation:**

Two classifiers:

| **Precisio** | **Recall n** |
| --- | --- |
| A 95% | 90% |
| B 98% | 85% |

Which is better? **You cannot say.** A is better on recall, B on precision. Now imagine your team runs eight experiments a day, each producing two numbers. Every comparison becomes a debate. Your iteration speed collapses.

F1 resolves it:

- A: 2 / (1/0.95 + 1/0.90) = **0.924**
- B: 2 / (1/0.98 + 1/0.85) = **0.910**

A wins. Decision made in one second, move on to the next experiment.

**Why harmonic mean rather than a plain average?** The harmonic mean punishes imbalance. A model with 100% precision and 1% recall has an arithmetic mean of 50.5% (looks decent!) but an F1 of 2%. F1 forces both to be reasonably good.

**Averaging across segments:** if you measure error separately by region,

| **Algorithm** | **US** | **China** | **India** | **Other** | **Averag e** |
| --- | --- | --- | --- | --- | --- |
| A | 3% | 7% | 5% | 9% | **6.0%** |
| B | 5% | 6% | 5% | 10% | **6.5%** |

then the average column makes the choice instant. (Consider a *weighted* average if the regions differ greatly in business value.)

**Real project example (CatShop fraud):** F1 is the standard, but think about whether it's really what you want. If a missed fraud costs ₹8,000 and a false alarm costs 2 minutes of a reviewer's time, recall matters far more than precision. Use **F-beta** (`F2` weights recall 2× more heavily) or build an explicit expected-cost

metric: `cost = 8000 × false_negatives + 50 × false_positives`. **Your metric should encode your actual business tradeoff, not the one that's conventional.**

#### 4.1.4 Satisficing and optimizing metrics

<a id="414-satisficing-and-optimizing-metrics"></a>

**The idea in one line:** When you care about several things, pick one to maximise and turn the rest into pass/fail thresholds.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Optimizing metric** | The one you push as far as possible. |
| **Satisficing metric** | One that just needs to clear a bar. Beyond the bar, more is worthless. |

**Deeper explanation:**

You care about accuracy *and* inference latency.

| **Classifier** | **Accuracy** | **Running time** |
| --- | --- | --- |
| A | 90% | 80 ms |
| B | 92% | 95 ms |
| C | 95% | 1500 ms |

You *could* combine them: `score = accuracy − 0.5 × time`. But that weighted sum is artificial, where does 0.5 come from? It implies you'd happily trade 1% accuracy for 2ms, which is almost certainly not true.

**The better framing:**

**Maximize accuracy***subject to***running time ≤ 100 ms**.

Accuracy is the optimizing metric. Running time is satisficing. C is disqualified regardless of its accuracy; among A and B, B wins. Clean, honest, and it matches how you actually think.

**The general rule: with N metrics, pick 1 to optimize and make the other N−1 satisficing.**

**The wakeword example** (Amazon Echo, Apple Siri, Google Home, Baidu DuerOS): maximize detection accuracy *subject to* at most 1 false positive every 24 hours. Nobody wants a device that wakes up randomly at 3am, and there is no accuracy gain that would make that acceptable, so it's a threshold, not a term in a sum.

**Real project example (CreatorRank):**

Maximize ROI prediction accuracy, subject to: inference under 200ms, model size under 500MB, and no demographic group's accuracy falling more than 5 points below the overall average.

That last one turns fairness into a hard constraint rather than an afterthought, which is exactly the right structural place for it.

#### 4.1.5 Train / dev / test distributions

<a id="415-train-dev-test-distributions"></a>

**The idea in one line:** Dev and test sets define your target, so they must reflect the data you actually care about, and they must match each other.

**Deeper explanation:**

You have cat images from the US, UK, other Europe, South America, India, China, other Asia, and Australia.

**Bad:** dev set from US/UK/Europe/South America, test set from India/China/Asia/Australia. You spend three months optimising for the first group and then get evaluated on the second. **You aimed at a moving target**: every improvement on the dev set may not transfer at all.

**Good:** shuffle all regions together, then split randomly into dev and test. Both come from the same distribution.

**The guideline, worth memorising:***choose a dev set and test set to reflect the data you expect to get in the future and consider important to do well on.*

Note the two conditions. Not just data you *have*, data you *expect*, and data you *care about*. If 90% of your current traffic is from a market you're about to exit, don't let it dominate your dev set.

**Real project example (CatShop):** The training set is mostly your 2023 catalogue. Your dev/test sets should reflect what you'll be selling *next quarter*, new categories, current photography styles, current seller mix. A dev set built from stale data optimises you for a business you no longer run.

#### 4.1.6 Size of dev and test sets

<a id="416-size-of-dev-and-test-sets"></a>

**The old rule:** 70/30 train/test, or 60/20/20 train/dev/test. Reasonable when your total dataset was 100 to 10,000 examples.

**The modern rule with 1,000,000+ examples:** 98% train / 1% dev / 1% test. One percent of a million is 10,000 examples, plenty for a reliable estimate, and you've freed up 380,000 examples for training that the old rule would have wasted.

**How to size the test set properly:** ask *"how many examples do I need to be confident in the result?"*, not *"what percentage is conventional?"*

The relevant intuition: to detect a 1% difference between two models with statistical confidence, you need on the order of 10,000 examples. To detect a 5% difference, a few hundred suffices. Size the test set to the smallest difference you actually need to resolve.

**Skipping the test set:** some teams use train/dev only, and call the dev set "test." Not generally recommended, you lose your unbiased estimate, and your reported number will be optimistic because you tuned against it. Acceptable if you understand and communicate that bias. Not acceptable if you're reporting the number to a customer or a board.

#### 4.1.7 When to change dev/test sets and metrics

<a id="417-when-to-change-devtest-sets-and-metrics"></a>

**The idea in one line:** If your metric ranks A above B but you'd rather ship B, your metric is wrong, fix the metric, not your judgment.

**Deeper explanation, Example 1:**

| **Algorithm** | **Error** | **Behaviour** |
| --- | --- | --- |
| A | 3% | Lets through pornographic images |
| B | 5% | Doesn't |

The metric says A. You and your users say B. **When the metric and your judgment disagree, and you're confident in your judgment, the metric is broken.**

**The fix, weighted error:**

Error = (1/Σw⁽ⁱ⁾) · Σ w⁽ⁱ⁾ · 1{ŷ⁽ⁱ⁾ ≠ y⁽ⁱ⁾} where w⁽ⁱ⁾ = 1 if x⁽ⁱ⁾ is non-pornographic 10 if x⁽ⁱ⁾ is pornographic

Now a single porn misclassification costs as much as ten ordinary ones, and the metric will correctly prefer B.

**The orthogonalization view, two separate steps, done separately:**

1. **Define a metric that captures what you want.** (Place the target.)
2. **Worry separately about how to do well on that metric.** (Aim and shoot.)

Don't muddle them. If you're arguing about how to weight porn images while also debating architecture choices, you're mixing two independent problems.

**Example 2, the distribution version:** you train and evaluate on high-quality downloaded images, but real users upload blurry, badly framed phone photos. Model A beats Model B on your dev set and loses in production. Same conclusion: **change the dev/test set** to real user images, and possibly the metric too.

**The general guideline:** if doing well on your current metric + dev/test set doesn't correspond to doing well on your actual application, change them.

**And a practical corollary:** it is better to have a *quick and imperfect* metric that lets you iterate than to spend a month designing the perfect one. Set something up, start moving, and change it when you discover it's misaligned. Discovering the misalignment requires having been running.

### Comparing to Human-Level Performance

<a id="comparing-to-human-level-performance"></a>

#### 4.1.8 Why human-level performance?

<a id="418-why-human-level-performance"></a>

**The idea in one line:** A human baseline tells you how much room is left, which is what turns a raw error number into an actionable diagnosis.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Human-level performance** | How well people do on the task. |
| **Bayes optimal error** | The theoretical best possible error for *any* model. Irreducible noise. |

**Deeper explanation:**

Plot accuracy against time. The typical shape: rapid progress up to human-level performance, then a distinct slowdown, then a slow asymptotic crawl toward Bayes optimal error, which is never surpassed, because it can't be. Some inputs are genuinely ambiguous. Some audio is genuinely too noisy to transcribe. That's the ceiling.

**Why progress slows once you pass human level, two concrete reasons:**

1. **Human-level performance is often close to Bayes error** on perception tasks. Humans are extremely good at vision and hearing. So passing humans means you're near the ceiling, and there's simply not much left to gain.
2. **The tools available below human level stop working above it.** While humans beat your model, you can: ask humans to label more data, manually inspect errors and ask *"why did a person get this right?"*, and analyse bias/variance against a known benchmark. Once your model is better than humans, none of these work, humans can't tell you why the model is wrong when they'd also be wrong.

#### 4.1.9 Avoidable bias

<a id="419-avoidable-bias"></a>

**The idea in one line:** Bias measured against the *achievable* floor, not against zero.

**Deeper explanation, two scenarios with identical training error:**

**Scenario A:**

|   | **Error** |
| --- | --- |
| Human | 1% |
| Training | 8% |
| Dev | 10% |

Gap from human to training = 7%. Gap from training to dev = 2%. → **Focus on bias.** Train a bigger network, train longer.

**Scenario B:**

|   | **Error** |
| --- | --- |
| Human | 7.5% |
| Training | 8% |
| Dev | 10% |

Gap from human to training = 0.5%. Gap from training to dev = 2%. → **Focus on variance.** Regularization, more data.

**Training error is 8% in both cases, and the correct next action is completely different.** That is the entire lesson. Without the human benchmark, you cannot tell these apart, and you have a 50% chance of spending the next month on the wrong thing.

**The two definitions:**

Avoidable bias = training error − Bayes (human-level) error Variance = dev error − training error

"Avoidable" is the operative word. In scenario B, 7.5% of the error is *unavoidable*, it's inherent to the problem. Trying to eliminate it would mean fitting noise, which is overfitting by definition.

**Human-level error is used as a proxy for Bayes error**, which you can never actually observe. It's an approximation, but usually a good one on perception tasks.

#### 4.1.10 Understanding human-level performance

<a id="4110-understanding-human-level-performance"></a>

**The idea in one line:** "Human-level" isn't one number, pick the right one for the purpose you have.

**The medical imaging example:**

| **Who** | **Error** |
| --- | --- |
| Typical untrained human | 3% |
| Typical doctor | 1% |
| Experienced doctor | 0.7% |
| Team of experienced doctors | 0.5% |

**Which is "human-level performance"?**

For **estimating Bayes error**: the purpose that matters for deciding what to work on, use **0.5%**, the best achievable. Bayes error is the *best possible*, so your proxy should be the best humans available, not the average.

For **publishing a paper** ("we beat doctors"), a different definition might be defensible. For **deployment decisions** ("is this safe to ship?"), you might compare against the typical doctor who would otherwise do the job. **The right definition depends on the purpose**: just be explicit about which one you're using and why.

**When the choice barely matters:**

|   | **Error** |
| --- | --- |
| Human (0.7% or 0.5%?) | 0.5–0.7% |
| Training | 5% |
| Dev | 6% |

Avoidable bias is ~4.3–4.5% either way; variance is 1%. Bias dominates regardless. The choice of benchmark doesn't change your decision, so don't spend a meeting on it.

**When the choice matters enormously:**

|   | **Error** |
| --- | --- |
| Human (0.7% or 0.5%?) | 0.5–0.7% |
| Training | 0.7% |
| Dev | 0.8% |

With 0.7% as the benchmark: avoidable bias = 0%, variance = 0.1% → work on variance. With 0.5% as the benchmark: avoidable bias = 0.2%, variance = 0.1% → work on bias.

Opposite conclusions. **The closer you get to human-level, the more the exact definition matters**: and this is precisely the regime where progress is hardest.

#### 4.1.11 Summary of bias/variance with human-level performance

<a id="4111-summary-of-biasvariance-with-human-level-performance"></a>

**The three-number diagnostic:**

Human (≈ Bayes) error ↕ ← AVOIDABLE BIAS Training error ↕ ← VARIANCE Dev error

**Compare the two gaps. The bigger one is your problem.**

**To reduce avoidable bias:**

- Train a bigger model
- Train longer / use better optimization (momentum, RMSprop, Adam)
- Better NN architecture / hyperparameter search
- Better features

**To reduce variance:**

- More data
- Regularization (L2, dropout, data augmentation)
- Better NN architecture / hyperparameter search

Notice that "NN architecture / hyperparameter search" appears in both lists, it's the one intervention that can help either way, which is also why it's a poor *first* move. Diagnose first, then choose the targeted tool.

**Real project example (CatShop, product categorisation):**

- Human accuracy: 98% (2% error)
- Training: 92% (8% error)
- Dev: 91% (9% error)

Avoidable bias 6%, variance 1%. **Bias problem, decisively.** Go deeper, use a pretrained backbone, train longer. Do *not* collect more data, it will not help, and it's the most expensive option on the menu. Three numbers just saved you a quarter's budget.

#### 4.1.12 Surpassing human-level performance

<a id="4112-surpassing-human-level-performance"></a>

**The idea in one line:** Once you beat humans, your compass stops working, you no longer know whether to attack bias or variance.

**Deeper explanation:**

|   | **Error** |
| --- | --- |
| Team of humans | 0.5% |
| One human | 1.0% |
| Training | 0.3% |

Dev 0.4%

Training error is *below* the best human benchmark. So what's your Bayes estimate? It could be 0.3%, 0.2%,

0. 1%, you have no way to know. **Avoidable bias becomes unmeasurable, and with it your ability to**

**choose a direction.** Progress becomes much slower and much more empirical.

**Where ML has significantly surpassed humans:**

- Online advertising (CTR prediction)
- Product recommendations
- Logistics (predicting transit time)
- Loan approvals

**The common thread:** all four involve **structured data**, **huge amounts of data**, and are **not natural perception tasks**. No human has ever looked at a hundred million click records and developed intuition about them. Machines have an inherent advantage where the task requires integrating far more data than a human can hold.

**Where humans still tend to excel:** natural perception, speech recognition in noisy conditions, some image recognition, medical tasks like reading ECGs or diagnosing skin cancer from photographs. Humans have had millions of years of evolution optimising vision and hearing. Though it's worth noting ML has now surpassed humans on several of these too, given enough labelled data.

**Real project example (CreatorRank):** Predicting campaign ROI from historical creator data is a *structured-data, no-human-intuition* task, exactly where ML should beat humans, and it usually does. But judging whether a creator's aesthetic *fits a brand* is a perception task where a good human marketer will beat your model for the foreseeable future. **Design your product around that split**: automate the structured prediction, keep humans in the loop on the perceptual judgment. That's a product-strategy decision falling directly out of an ML-strategy insight.

## Structuring ML Projects, Part II

<a id="structuring-ml-projects-part-ii"></a>

*Four themes: Error Analysis, Mismatched Distributions, Learning from Multiple Tasks, End-to-End Deep Learning.*

### 1. Error Analysis

<a id="1-error-analysis"></a>

#### 1.1 Carrying out error analysis

<a id="11-carrying-out-error-analysis"></a>

**The idea in one line:** Before spending a month fixing a problem, spend an hour counting how much of your error that problem actually causes.

**Deeper explanation:**

Your cat classifier has 10% error. Someone notices it's misclassifying dogs as cats and proposes a dog-specific fix. That's months of work. Should you do it?

**Do the error analysis instead, it takes an hour:**

1. Pull ~100 mislabelled dev-set examples.
2. Count how many are dogs.
3. Do the arithmetic.

**Case A:** 5 of 100 are dogs. Fixing dogs *perfectly* takes you from 10% error to 9.5%. **A 5% relative improvement for months of work.** Don't do it.

**Case B:** 50 of 100 are dogs. Fixing dogs perfectly takes you from 10% to 5%, **halving your error.** Absolutely do it.

Same idea, same effort, wildly different value. The one-hour analysis told you which world you're in. This is the highest return-on-time activity in applied machine learning, and it is chronically skipped because it feels less like real work than training a model.

**The concept has a name: the ceiling on improvement.** The error category's share of total error is the *most* you can gain by fixing it completely.

#### 1.2 Evaluating multiple ideas in parallel

<a id="12-evaluating-multiple-ideas-in-parallel"></a>

**The idea in one line:** Build a spreadsheet, one row per misclassified example, one column per error category, and let the tallies pick your roadmap.

| **Image** | **Dog** | **Great cat** | **Blurry** | **Instagram filter** | **Comments** |
| --- | --- | --- | --- | --- | --- |
| 1 | ✓ |   |   |   | pitbull |
| 2 |   |   | ✓ | ✓ |   |
| 3 |   | ✓ | ✓ |   | lion, rainy day |
| ... |   |   |   |   |   |
| **% of total** | **8%** | **43%** | **61%** | **12%** |   |

*(Percentages exceed 100% because an image can belong to several categories.)*

**Reading the result:** blurry images cause 61% of errors, great cats 43%, dogs only 8%. The roadmap writes itself, work on blur and large cats; ignore the dog idea entirely.

**Practical notes:**

- **Add new categories mid-way** as you spot patterns you didn't anticipate. That's normal, go back and re-score the earlier rows.
- **The comments column is where the insight lives.** "rainy day," "pitbull," "photo taken through glass", these become new categories.
- 100 examples is usually enough for percentages accurate to a few points. That's all the precision you need to prioritise.

**Real project example (CreatorRank):** Your ROI prediction is off by more than 30% on 22% of campaigns. Pull 100 of the worst and categorise them: 41% are creators with fewer than 5,000 followers (thin data), 30% had a viral post distorting their averages, 18% were in categories with almost no training data, 11%

miscellaneous. **Your roadmap is now written, and it's about data coverage and outlier handling, not about the model architecture you were about to redesign.**

#### 1.3 Cleaning up incorrectly labelled data

<a id="13-cleaning-up-incorrectly-labelled-data"></a>

**The idea in one line:** Neural networks tolerate random label noise well but systematic label noise badly, so measure which kind you have before cleaning.

**Deeper explanation:**

**Random errors in the training set:** deep learning is fairly robust to these. If 3% of your labels are wrong in random directions, the errors roughly cancel out and the model still learns the true pattern. Usually not worth fixing.

**Systematic errors:** deep learning is *not* robust to these. If every white dog was labelled "cat," the model will confidently and consistently learn that white dogs are cats. There's no cancellation, the noise all points the same way. **These must be fixed.**

**How to decide, add a column to your error analysis table:**

| **Image DogGreat cat** | **Blurry Incorrectly labelled** |
| --- | --- |
| ... | ✓ |
| **% of total** 8% 43% | 61% **6%** |

Now compare 6% against your total dev error:

| **Overall dev error** | **From bad labels** | **From other causes** | **Verdict** |
| --- | --- | --- | --- |
| 10% | 0.6% | 9.4% | Ignore the labels. Work on real errors. |
| 2% | 0.6% | 1.4% | **Label noise is now ~30% of your error. Fix it.** |

Same 6%, opposite conclusions. **As your model improves, label noise becomes a larger fraction of what remains**: so a labelling issue you correctly ignored six months ago may now be your top priority. Re-run the analysis periodically.

**Guidelines for correcting dev/test sets:**

1. **Apply the same correction process to dev and test together**, so they stay from the same distribution. Fixing only the dev set breaks the pairing.
2. **Also examine examples the model got *right*.** If the model predicted "cat" and the label said "cat" but it's actually a dog, that's an error you never counted. This is harder and often skipped, but skipping it means your accuracy estimate is optimistically biased.
3. **Train and dev/test may now differ slightly** (since you probably won't relabel a million training examples). That's usually fine, the dev/test alignment is what matters most.

**Real project example (CatShop):** Your product categories were labelled by three contractors. Sample 200 and check inter-annotator agreement. If two labellers disagree 15% of the time, your model's ceiling is

roughly that agreement rate, no model can beat inconsistent ground truth. **Fix the labelling guidelines before you touch the model.** This is one of the most common and most invisible reasons real projects stall.

#### 1.4 Build your first system quickly, then iterate

<a id="14-build-your-first-system-quickly-then-iterate"></a>

**The idea in one line:** Ship a bad version fast, then let error analysis tell you what to improve, because your guesses about what matters are usually wrong.

**Deeper explanation, the speech recognition example.** Directions you could pursue:

- Noisy background
- Accented speech
- Far-field microphones
- Young children's speech
- Stuttering, filler words

Every one is a legitimate multi-month research programme. Which do you pick? **You genuinely don't know yet**, and neither does anyone else on the team, however confidently they argue.

**The guideline:**

1. Set up dev/test sets and a metric. *(Decide where the target is.)*
2. Build a quick-and-dirty first system. *(Days, not months.)*
3. Use bias/variance analysis and error analysis to decide where to invest next. *(Let the data direct you.)*

The first system is not meant to be good. **It's meant to be a measuring instrument.** Without it you have opinions; with it you have error categories with percentages.

**The important exception:** if you're working in a mature area with strong literature, standard image classification, standard speech recognition, building on existing designs is entirely reasonable. Don't reinvent ResNet. This advice is for genuinely new problems.

**Real project example (SaaSChurn):** Week 1 target, logistic regression on 10 obvious features, a dev set, an AUC number, a spreadsheet of the 50 worst predictions. That's it. Week 2 is then informed by evidence instead of a whiteboard debate. Teams that skip week 1 and spend a month designing the "right" architecture routinely discover in week 5 that their real problem was a data pipeline bug.

### 2. Mismatched Training and Dev/Test Sets

<a id="2-mismatched-training-and-devtest-sets"></a>

#### 2.1 Training and testing on different distributions

<a id="21-training-and-testing-on-different-distributions"></a>

**The idea in one line:** Train on whatever data you can get; evaluate only on data that looks like production.

**The cat app example:**

- 200,000 high-quality web images (easy to scrape)
- 10,000 blurry user-uploaded photos (what you actually care about)

**Option 1 (bad):** shuffle all 210,000 and split randomly. Your dev set is then ~95% web images and ~5% user photos. **You're optimising for web images**, which is not your product. Your dev-set improvements won't transfer.

**Option 2 (good):**

- Train: 200,000 web + 5,000 user
- Dev: 2,500 user only
- Test: 2,500 user only

The dev/test sets now reflect reality. You still get to use all 200,000 web images for training, they contain plenty of useful signal about what cats look like, but they no longer distort your target.

**The speech recognition version:** training data is purchased/scraped speech from various sources; dev and test come from actual in-car voice queries. Same principle: **aim the dev/test set at the distribution you want to win on, even when the training data comes from elsewhere.**

**Real project example (CreatorRank):** Train on 500,000 creators with cheap proxy labels (engagement-rate-based). Dev and test on 2,000 creators with real measured campaign ROI. The proxy labels are abundant and noisy; the real labels are scarce and accurate. **Use abundance for learning and accuracy for evaluation.** This is a standard, sensible pattern in industry.

#### 2.2 Bias and variance with mismatched data distributions

<a id="22-bias-and-variance-with-mismatched-data-distributions"></a>

**The idea in one line:** When train and dev come from different distributions, a train→dev gap could be overfitting *or* distribution shift, and the fix for each is completely different. The training-dev set tells them apart.

**Jargon decoded:** the **training-dev set** is a slice held out from the *training* distribution and never trained on. Same distribution as training data, unseen like dev data.

**Deeper explanation:**

Training error 1%, dev error 10%. Is that variance, or data mismatch? You cannot tell, because two things changed at once: the data is now unseen *and* it's from a different distribution.

**The fix, introduce a fourth set:**

Training set ← model trains on this Training-dev set ← same distribution as training, NOT trained on Dev set ← target distribution Test set ← target distribution

Now read the gaps:

**TrainingTraining-deDevDiagnosis**

**v**

1% 9% 10% **Variance.** Error jumped on unseen data of the *same* distribution → overfitting.

| 1% | 1.5% | 10% | **Data mismatch.** Generalises fine within its own distribution; the problem is the distribution *change*. |
| --- | --- | --- | --- |
| 10% | 11% | 12% | **Avoidable bias.** Can't even fit the training data. |
| 10% | 11% | 20% | **Bias AND data mismatch.** |

One extra data split converts an ambiguous symptom into a precise diagnosis.

#### 2.3 The general formulation

<a id="23-the-general-formulation"></a>

**Four levels, four named gaps:**

Human / Bayes error ↕ avoidable bias Training error ↕ variance Training-dev error ↕ data mismatch Dev error ↕ degree of overfitting to the dev set Test error

**On that last gap:** if dev error is 10% and test error is 15%, you have overfit the dev set, you've tuned against it so many times that its score no longer generalises. The fix is a bigger dev set, or fewer decisions made against it.

**Real project example, a full diagnosis:**

| **Set** | **Error** |
| --- | --- |
| Human | 4% |
| Training | 7% |
| Training-dev | 10% |
| Dev | 12% |
| Test | 12% |

Avoidable bias 3%, variance 3%, data mismatch 2%, dev overfitting 0%. Three roughly equal problems and no dominant one, so attack them in order of cost. Bias is usually cheapest (bigger network, train longer), so start there. And note the healthy sign: dev and test match, so your evaluation is trustworthy.

#### 2.4 Addressing data mismatch

<a id="24-addressing-data-mismatch"></a>

**The idea in one line:** There's no clean algorithm for this, do manual error analysis to find *how* the distributions differ, then close the gap deliberately.

**The process:**

1. **Carry out manual error analysis** comparing training and dev/test data. Listen to the audio. Look at the images. What's actually different? Common findings: dev data is noisier, has car noise, has different accents, has different lighting, has more compression artifacts.
2. **Make the training data more similar** to dev/test, for example, add simulated car noise to your clean training audio.
3. **Or collect more data that resembles dev/test**: go record actual in-car audio.

**Honest note:** unlike bias and variance, there is no systematic recipe here. It requires looking at data with your own eyes. That's uncomfortable for engineers who'd rather write code, and it's exactly why teams that do it have an advantage.

#### 2.5 Artificial data synthesis

<a id="25-artificial-data-synthesis"></a>

**The idea in one line:** You can manufacture realistic training data, but you'll probably synthesise a much narrower slice of reality than you think.

**The speech example:** you have 10,000 hours of clean speech and 1 hour of recorded car noise. Overlay them to create 10,000 hours of realistic in-car audio. It sounds perfect to a human ear.

**The trap:** you looped that *same 1 hour* of car noise 10,000 times. The model can memorise that specific noise pattern and learn to subtract exactly it, while remaining helpless against any other car noise. **To a human the synthesised data sounds like 10,000 hours of variety. To the model it's 1 hour of noise repeated.**

**The car recognition version:** you generate training images with computer graphics. Perhaps 20 car models rendered from many angles. Visually convincing. But there are thousands of real car models, so you've synthesised a tiny subset of the true distribution and the model will overfit to your 20 shapes.

**The general principle, and it's subtle:***synthetic data can look convincing to humans while covering a vanishingly small slice of the true distribution.* Human perceptual judgment is a bad guide to distributional coverage. Deliberately maximise the *diversity* of your synthesis, not just its realism, 100 hours of varied noise beats 1 hour looped, even if each individual sample sounds no better.

**Real project example (CatShop):** Augmenting product photos with synthetic backgrounds using 10 background images will teach the model those 10 backgrounds. Use hundreds, sampled from real photography, or use a generative model to produce genuinely varied ones. And always keep a held-out set of **real** examples to measure whether the synthesis is helping or just teaching the model your synthesiser's quirks.

### 3. Learning from Multiple Tasks

<a id="3-learning-from-multiple-tasks"></a>

#### 3.1 Transfer learning

<a id="31-transfer-learning"></a>

**The idea in one line:** Take a network trained on a big task, keep everything it learned about low-level structure, and retrain only the last bit for your small task.

**Jargon decoded:**

| **Term** | **Plain meaning** |
| --- | --- |
| **Transfer learning** | Reusing a model trained on task A as the starting point for task B. |
| **Pre-training** | The original training on the big dataset. |
| **Fine-tuning** | Additional training on your smaller task-specific dataset. |
| **Freezing layers** | Holding some layers' weights fixed so they don't update. |

**The mechanics:**

1. Take a network trained on task A (say, ImageNet image recognition, 14 million images, 1000 classes).
2. Delete the last layer (or last few).
3. Add a new final layer with randomly initialised weights, sized for task B (say, 3 classes of radiology finding).
4. Train on task B's data.

**How much to retrain depends on how much task-B data you have:**

| **Task B data** | **Approach** |
| --- | --- |
| Very small (100s) | **Freeze everything**, retrain only the new final layer. |
| Small (1,000s) | Freeze early layers, retrain the last few. |
| Medium (10,000s) | Retrain the whole network at a low learning rate. |
| Large (100,000s+) | Retrain everything, or consider training from scratch. |

**Why it works so well:** the early layers of a vision network learn edges, corners, textures, and simple shapes. **Those are useful for essentially any image task.** There's no reason to relearn "what an edge looks like" from your 500 X-rays when a network has already learned it from 14 million photographs. You're transferring generic visual competence and only learning the task-specific part.

**When transfer learning makes sense, all three should hold:**

1. Task A and task B have the **same input type** (both images, both audio, both text).
2. You have **far more data for A than for B**. (The reverse makes no sense, why transfer *from* the smaller dataset?)
3. **Low-level features from A are plausibly useful for B.**

**Real project example (CatShop):** For product categorisation with 8,000 images, do *not* train from scratch. Load a pretrained ResNet-50 or EfficientNet, replace the final layer with 47 outputs, freeze the backbone, train the head for a few epochs, then unfreeze the top third and fine-tune at a very low learning rate (1e-5). This routinely takes you from ~70% accuracy (from scratch) to ~92%, with less compute. **Transfer learning is the single highest-leverage technique available to a small team.** The same is true in NLP, fine-tuning a pretrained language model rather than training one.

#### 3.2 Multi-task learning

<a id="32-multi-task-learning"></a>

**The idea in one line:** Train one network to do several related things at once, so each task benefits from the others' data.

**The autonomous driving example:**

A single image may contain pedestrians, cars, stop signs, *and* traffic lights simultaneously. So the label isn't one class, it's a vector:

y⁽ⁱ⁾ = [0, 1, 1, 0] # no pedestrian, car yes, stop sign yes, no traffic light

**This is fundamentally different from softmax.** Softmax picks one of C classes, forcing probabilities to sum to 1. Here, **each example can carry several positive labels at once**, and the labels don't compete.

**Architecture:**

- One shared network body.
- 4 output units, each with a **sigmoid** (not softmax).
- Loss = sum of 4 independent binary cross-entropies.

```text
J = (1/m) · Σᵢ Σⱼ L(ŷⱼ⁽ⁱ⁾, yⱼ⁽ⁱ⁾)
```

**A genuinely useful property: it still works with missing labels.** If some images were labelled only for cars and not for pedestrians, just **sum the loss over the labels that are actually present** and skip the rest. You don't need complete labels on every example, which is enormous in practice, since complete labelling is expensive and rare.

**When multi-task learning makes sense:**

1. **The tasks share useful low-level features.** Detecting cars and detecting pedestrians both need edges, shapes, and textures.
2. **Each task has roughly a similar amount of data**, and the combined data from other tasks is much larger than any single task's data. If task 1 has 1,000 examples and tasks 2–100 have 1,000 each, task 1 effectively gains 99,000 examples' worth of feature learning.
3. **You can train a big enough network.** With sufficient capacity, multi-task learning rarely hurts and usually helps. With *insufficient* capacity, the tasks compete for parameters and multi-task learning can be worse than separate models. Capacity is the deciding factor.

**Practical reality:** multi-task learning is used **far less often than transfer learning**. Object detection is the main exception, where it's standard. Transfer learning is easier, needs no simultaneous labelling, and usually delivers more.

**Real project example (CreatorRank):** One network predicting engagement rate, campaign ROI, audience-brand fit, and content-safety risk from the same creator features. All four depend on overlapping underlying signals (audience quality, content consistency, posting patterns), so sharing a body helps, especially for content-safety risk, where you have very few labelled examples and would struggle to train a standalone model.

### 4. End-to-End Deep Learning

<a id="4-end-to-end-deep-learning"></a>

#### 4.1 What is end-to-end deep learning?

<a id="41-what-is-end-to-end-deep-learning"></a>

**The idea in one line:** Replace a hand-designed multi-stage pipeline with one network mapping raw input straight to final output.

**The speech recognition example:**

**Traditional pipeline:**

audio → hand-engineered features (MFCC) → phonemes → words → transcript

Each stage is separately designed, separately tuned, often by different specialists. "Phonemes" are a linguistics concept that humans invented.

**End-to-end:**

audio → [one big neural network] → transcript

**The critical condition, data volume:**

| **Training data** | **Which wins** |
| --- | --- |
| 3,000 hours | Traditional pipeline |
| 10,000 hours | Roughly comparable |
| 100,000 hours | End-to-end wins clearly |

End-to-end tends to beat hand-designed pipelines **only when you have a large amount of data.** With modest data, the traditional pipeline usually wins, because the hand-designed structure is itself a form of injected knowledge that substitutes for data you don't have.

#### 4.2 The face recognition (turnstile) example

<a id="42-the-face-recognition-turnstile-example"></a>

**The task:** an employee walks toward a turnstile; a camera should identify them.

**End-to-end approach:** raw camera image → identity. **This works poorly.**

**Two-step approach, and it works much better:**

1. **Detect and crop the face.** The person could be near or far, left or right; the image contains lots of irrelevant background.
2. **Compare the cropped face against the employee database.** A specialised sub-problem: "are these two faces the same person?"

**Why the pipeline wins here:** each sub-task has plenty of data available. There are huge public datasets for face detection, and huge datasets of face pairs labelled same/different. But there is *no* dataset of "raw turnstile camera images → employee identity" at scale, you'd need millions of images of your specific

employees at your specific turnstile. **The pipeline lets you exploit data that exists; end-to-end requires data that doesn't.**

#### 4.3 More examples

<a id="43-more-examples"></a>

**Machine translation, end-to-end wins.** English→French sentence pairs are abundant (millions from parliamentary proceedings, subtitles, websites). The old pipeline (parse → align → transfer → generate) has been thoroughly beaten. This is end-to-end's clearest victory.

**Estimating a child's age from a hand X-ray, the pipeline wins.** The traditional approach segments the bones, measures their lengths, and looks up the age in a paediatric reference table. End-to-end (X-ray → age) fails because you'd need an enormous dataset of X-rays with known ages, and it doesn't exist. Meanwhile the pipeline's stages each have data, and the lookup table encodes decades of medical knowledge for free.

**The pattern across all three:** ask not "which is more elegant" but **"which stages do I have data for?"**

#### 4.4 Pros and cons of end-to-end deep learning

<a id="44-pros-and-cons-of-end-to-end-deep-learning"></a>

**Pros:**

1. **It lets the data speak.** The pipeline forces human preconceptions onto the model. Phonemes are a *human linguistic construct*, there's no guarantee they're the optimal intermediate representation for a machine. End-to-end learning discovers whatever representation actually minimises error, which may be something no linguist would have proposed.
2. **Less hand-designing of components.** Less specialist labour, less brittle engineering, fewer stages to maintain and debug.

**Cons:**

1. **Needs a large quantity of end-to-end labelled data.** Not data for the sub-tasks, data for the *whole* mapping. That's the expensive kind, and often it simply doesn't exist.
2. **It excludes potentially useful hand-designed components.** Hand-designed components are a way of injecting human knowledge into the system. When data is scarce, that knowledge is valuable, sometimes the only thing keeping the system viable.

**The central framing:** hand-designed components and data are **substitutes for each other**. Both are ways of getting knowledge into the model. With little data, human knowledge carries the load. With lots of data, human knowledge starts to *constrain* rather than help.

#### 4.5 Whether to use end-to-end deep learning

<a id="45-whether-to-use-end-to-end-deep-learning"></a>

**The key question, and really the only one that matters:**

**Do you have enough data to learn the complexity of the full input→output mapping?**

If yes, end-to-end is likely better. If no, use a pipeline and let hand-designed components carry the knowledge your data can't.

**The autonomous driving example:** raw image → steering angle is theoretically end-to-end. But nobody does it that way for production systems, because you'd need an astronomical amount of data covering every rare and dangerous situation, and those are precisely the situations where you have least data and can least afford a mistake. The practical approach is a pipeline: detect cars and pedestrians (huge datasets available) → plan a route (motion planning, well-studied) → compute steering (control theory, solved decades ago). Deep learning does the perception step, where it's strongest; classical methods do the planning and control, where they're proven and verifiable.

**Real project example (CreatorRank):** "Brand brief → ranked list of creators" is superficially an end-to-end problem. Don't build it that way. You have maybe 2,000 historical campaigns, nowhere near enough. Build a pipeline: extract brand attributes from the brief (an LLM, or rules) → filter the creator pool by hard constraints (audience geography, category, follower range) → score the remaining candidates on fit (a small trained model) → rank. Each stage is independently testable, independently improvable, and independently explainable to a client. **When you have thousands rather than millions of examples, pipelines aren't a compromise, they're the correct engineering choice.**

### Quick recap of the four themes

<a id="quick-recap-of-the-four-themes"></a>

- **Error analysis**: spend an hour counting error types before spending a month fixing one.
- **Data mismatch**: set dev/test to the distribution you care about, and use a training-dev set to separate variance from mismatch.
- **Multiple tasks**: transfer learning when the target task is data-poor; multi-task learning when many related tasks share features.
- **End-to-end**: powerful with lots of data, but pipelines remain better when data for the full mapping is limited.

## Appendix A: Complete Jargon Glossary

<a id="appendix-a-complete-jargon-glossary"></a>

| **Term** | **Plain meaning** |
| --- | --- |
| **Activation** | The output of a neuron after the non-linear function. `a = g(z)` |
| **Activation function** | The non-linear squashing function: ReLU, sigmoid, tanh, softmax. |
| **Adam** | Adaptive Moment estimation. Momentum + RMSprop + bias correction. The default optimizer. |
| **AUC / ROC-AUC** | Area under the ROC curve. A threshold-independent measure of ranking quality. 0.5 = random, 1.0 = perfect. |
| **Autograd / automatic differentiation** | Frameworks computing gradients for you from the forward computation. |
| **Avoidable bias** | Training error minus Bayes (human-level) error. The bias you can actually do something about. |
| **Backpropagation** | Computing gradients by applying the chain rule backward through the network. |
| **Batch gradient descent** | Using the whole training set for one update. |
| **Batch normalization** | Normalizing layer pre-activations across a mini-batch, with learnable γ and β. |
| **Bayes optimal error** | The best error any model could ever achieve. Irreducible. |
| **Bias (parameter)** | The constant `b` added inside a neuron. |
| **Bias (statistical)** | Underfitting. Model too simple. |
| **Bias correction** | Dividing an exponentially weighted average by (1 − βᵗ) to fix its early-iteration underestimate. |
| **Binary classification** | Two-class prediction. |
| **Broadcasting** | NumPy automatically expanding array shapes to make an operation work. |
| **Cache** | Values stored during forward prop for reuse in backward prop. |
| **CNN** | Convolutional Neural Network. For images and spatial data. |
| **Convex** | Bowl-shaped cost function with a single global minimum. |
| **Cost function J** | Average loss over all training examples. The thing you minimise. |
| **Covariate shift** | The input distribution changes while the input→output mapping stays the same. |
| **Cross-entropy** | The standard classification loss. Also called log loss. |
| **Data augmentation** | Creating new training examples by label-preserving transformations. |
| **Data mismatch** | Training and dev/test data coming from different distributions. |
| **Dead ReLU** | A neuron permanently outputting zero with zero gradient. |
| **Deep network** | A network with several hidden layers. |
| **Dev set** | Validation set. Used for tuning and model comparison. |
| **Dropout** | Randomly zeroing units during training as regularization. |
| **Early stopping** | Halting training when dev error stops improving. |
| **Epoch** | One full pass through the training set. |
| **Exploding gradients** | Gradients growing exponentially with depth. Causes NaN. |
| **Exponentially weighted average** | Smoothed running average with exponentially decaying weights on the past. |
| **F1 score** | Harmonic mean of precision and recall. |
| **Fine-tuning** | Retraining a pretrained model on your specific task. |
| **Forward propagation** | Computing the prediction from inputs, layer by layer. |
| **Frobenius norm** | Sum of squares of all elements in a matrix. Used in L2 regularization. |
| **Gradient** | The derivative. The slope of the cost with respect to a parameter. |
| **Gradient clipping** | Capping gradient magnitude to prevent explosion. |
| **He initialization** | `randn * sqrt(2/n_prev)`. For ReLU networks. |
| **Hidden layer** | Any layer between input and output. |
| **Hyperparameter** | A setting you choose before training (learning rate, layers, λ). |
| **Inverted dropout** | The standard dropout implementation, rescaling by keep_prob during training. |
| **keep_prob** | Probability of keeping a unit under dropout. |
| **L1 regularization** | Penalty on the sum of absolute weights. Produces sparsity. |
| **L2 regularization** | Penalty on the sum of squared weights. The standard. Also called weight decay. |
| **Layer normalization** | Normalizing across features within one example. Batch-size independent. |
| **Learning rate α** | Step size in gradient descent. The most important hyperparameter. |
| **Learning rate decay** | Reducing α over the course of training. |
| **Logits** | Raw pre-softmax/pre-sigmoid scores. |
| **Loss function L** | Error on a single example. |
| **Mini-batch** | A subset of the training data used for one update. |
| **Momentum** | Using a smoothed average of past gradients as the update direction. |
| **Multi-task learning** | One network trained on several related tasks simultaneously. |
| **Normalization** | Rescaling data to mean 0, variance 1. |
| **One-hot encoding** | Representing a class as a vector with a single 1. |
| **Optimizing metric** | The metric you maximise. |
| **Orthogonalization** | Designing so each control affects exactly one outcome. |
| **Overfitting** | High variance. Great on train, poor on dev. |
| **Parameters** | W and b. Learned by gradient descent. |
| **Precision** | Of predicted positives, the fraction that were truly positive. |
| **Recall** | Of actual positives, the fraction you found. |
| **ReLU** | max(0, z). The default hidden-layer activation. |
| **Regularization** | Any technique that reduces overfitting. |
| **RMSprop** | Root Mean Square propagation. Per-parameter adaptive step sizes. |
| **RNN** | Recurrent Neural Network. For sequences. |
| **Saddle point** | Zero gradient, but curving up in some directions and down in others. |
| **Satisficing metric** | A metric that just needs to clear a threshold. |
| **Saturation** | An activation in its flat region, where the gradient is near zero. |
| **Sigmoid** | 1/(1+e⁻ᶻ). Range (0,1). For binary output layers. |
| **Softmax** | Turns C logits into C probabilities that sum to 1. |
| **Symmetry breaking** | Random initialization so neurons learn different things. |
| **Tanh** | Zero-centred sigmoid variant. Range (−1,1). |
| **Test set** | Used once, at the end, for an unbiased performance estimate. |
| **Training-dev set** | Held out from the training distribution, not trained on. Separates variance from data mismatch. |
| **Transfer learning** | Reusing a model trained on a large task for a smaller related one. |
| **Underfitting** | High bias. Model too simple. |
| **Vanishing gradients** | Gradients shrinking exponentially with depth. Early layers stop learning. |
| **Variance (statistical)** | Overfitting. |
| **Vectorization** | Replacing loops with whole-array operations. |
| **Weight decay** | Another name for L2 regularization. |
| **Xavier / Glorot initialization** | `randn * sqrt(1/n_prev)`. For tanh/sigmoid networks. |

## Appendix B: Decision Cheat Sheets

<a id="appendix-b-decision-cheat-sheets"></a>

### B.1 Which activation function?

<a id="b1-which-activation-function"></a>

| **Layer** | **Situation** | **Use** |
| --- | --- | --- |
| Hidden | Default | **ReLU** |
| Hidden | Dead units observed | **Leaky ReLU** |
| Hidden | Transformer / modern | **GELU** |
| Output | Binary classification | **Sigmoid** |
| Output | Multi-class, pick one | **Softmax** |
| Output | Multi-label, several true | **Sigmoid on each** |
| Output | Regression, any real value | **Linear (none)** |
| Output | Regression, non-negative | **ReLU** |

### B.2 My model isn't working, what do I check?

<a id="b2-my-model-isnt-working-what-do-i-check"></a>

| **Symptom** | **Likely cause** | **Fix** |
| --- | --- | --- |
| Cost is NaN | Exploding gradients, `log(0)`, α too large | Lower α, gradient clipping, add ε inside logs |
| Cost doesn't decrease at all | α too small; data not normalised; bad init; bug | Print the cost every iteration; sanity-check shapes; try α ×10 |
| Cost decreases then flatlines early | Plateau, vanishing gradients | Switch to Adam; He init; add BatchNorm |
| Train error high | **High bias** | Bigger network, train longer, Adam, better features |
| Train low, dev high | **High variance** | More data, dropout, L2, augmentation |
| Train ≈ dev ≈ human | You're done | Ship it |
| Dev good, production bad | Data mismatch or leakage | Rebuild dev set from production data; check for time leakage |
| Test error ≫ dev error | Overfit the dev set | Bigger dev set; fewer decisions made against it |
| Great offline, useless in production | Metric doesn't match the business goal | Change the metric |

Model can't beat a Bug, leakage, or genuinely Verify a shuffled-label test gives trivial baseline no signal chance performance

### B.3 First 60 minutes on any new problem

<a id="b3-first-60-minutes-on-any-new-problem"></a>

1. Define the business outcome you want to improve, in a sentence.
2. Choose the metric. One optimizing metric, the rest satisficing.
3. Build dev/test sets from the distribution you actually care about. Same distribution as each other. Split by time if time matters.
4. Establish a **human-level baseline** if one exists.
5. Build the dumbest possible model (logistic regression, or predict the majority class). Record the number.
6. Only now start improving, and after every run, compare train error to human error, and dev error to train error.

### B.4 Sensible starting hyperparameters

<a id="b4-sensible-starting-hyperparameters"></a>

| **Hyperparameter** | **Start here** |
| --- | --- |
| Optimizer | Adam (or AdamW) |
| Learning rate | 0.001 |
| β₁, β₂, ε | 0.9, 0.999, 1e-8 (don't touch) |
| Batch size | 64 or 128 |
| Weight init | He (ReLU) / Glorot (tanh) |
| Hidden activation | ReLU |
| Layers | 2–3 for tabular, pretrained backbone for images |
| Dropout | 0 initially, add only if overfitting, then 0.2–0.5 |
| L2 λ | 0 initially, then try 0.001–0.1 |
| LR schedule | ReduceLROnPlateau(factor=0.5, patience=5) |
| Early stopping | patience=10, restore_best_weights=True |

## Appendix C: A Suggested Practice Path

<a id="appendix-c-a-suggested-practice-path"></a>

Reading these notes builds vocabulary. Building things builds understanding. A reasonable sequence:

1. **Implement logistic regression in pure numpy**, vectorized, on a small binary dataset. No frameworks. You should be able to write the five lines of section 1.1.12 from memory afterwards.
2. **Implement a 2-layer network in pure numpy**, including backprop by hand. This is the single most useful exercise in the whole curriculum. Do it once, properly. Then never again.
3. **Implement an L-layer network** with a loop over layers. Verify your gradients numerically (compute `(J(θ+ε) − J(θ−ε)) / 2ε` and compare to your analytic gradient).
4. **Rebuild the same thing in PyTorch** in 20 lines. Appreciate what the framework is doing for you.
5. **Take a real dataset**: one from your own domain, ideally, and run the full Chapter 4 loop: metric, dev/test split, dumb baseline, bias/variance diagnosis, error analysis on 100 examples, targeted improvement. Do this end to end before optimising anything.
6. **Fine-tune a pretrained model** on a small image or text dataset. Observe how much better it is than training from scratch.
7. **Deploy something.** Even badly. The gap between a notebook and a served model contains lessons that no course covers.

The one habit that separates people who get good at this from people who don't: **after every training run, write down what you changed, what happened, and what you concluded.** Three lines in a file. Do it for fifty runs and you'll have built the intuition that no amount of reading provides.
