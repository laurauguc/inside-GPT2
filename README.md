# Inside GPT-2: Following a Prompt Through the Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laurauguc/inside-GPT2/blob/main/Interpreting%20GPT-2%20Parameters.ipynb)

A complete walkthrough of GPT-2 inference that independently reconstructs the model's forward pass from its learned parameters, verifying every intermediate result against the official Hugging Face implementation.

The entire walkthrough is available as an interactive Google Colab notebook, allowing readers to execute every computation themselves.
---

## Overview

Most Transformer tutorials either explain the architecture at a high level or focus on training. This repository instead answers a different question:

> **What exactly happens inside GPT-2 when it predicts the next token?**

Using the single prompt

> **"The cat is chasing a"**

we independently reproduce the complete forward pass of GPT-2 Small.

Rather than relying on Hugging Face's implementation of the Transformer layers, we apply the underlying mathematical operations directly—matrix multiplications, layer normalization, self-attention, MLPs, and residual connections—using only the model's learned parameters and the hidden states produced by previous layers.

After every stage, we verify that the resulting tensors exactly match the corresponding outputs from the official implementation.

By the end of the walkthrough, every value contributing to GPT-2's prediction has been traced from the model's weights to the generated token.

---

## Walkthrough

Beginning with tokenization, we follow the prompt

> **"The cat is chasing a"**

through every stage of GPT-2 Small.

The model consists of:

- Tokenization
- Token embeddings
- Positional embeddings
- **12 identical Transformer blocks**
- Final layer normalization
- Projection to vocabulary logits
- Next-token prediction

Each of the 12 Transformer blocks has the same architecture:

1. Pre-attention layer normalization
2. Multi-head self-attention
   - Query, key, and value projections
   - Attention score computation
   - Causal masking
   - Softmax normalization
   - Context vector computation
   - Head concatenation
   - Output projection
3. Residual connection
4. Pre-MLP layer normalization
5. Feed-forward (MLP) network
6. Residual connection

GPT-2 Small contains **12 Transformer blocks** with identical architectures but different learned parameters.

Because every block performs the same sequence of operations, the walkthrough independently reconstructs and verifies the computations of the **first block** in detail. The remaining blocks are then executed using the official Hugging Face implementation, allowing us to follow the hidden representations through the entire network without repeatedly implementing the same algorithm.

Each section follows the same structure:

1. Introduce the mathematical operation.
2. Explain the intuition behind the computation.
3. Implement the operation directly using GPT-2's learned parameters.
4. Verify the intermediate results against the official implementation.
5. Discuss what the transformation represents and why it is useful.

Following the same process for every layer builds a complete understanding of how information flows through the Transformer.

---

## Why a single running example?

Every chapter follows the same prompt:

> **"The cat is chasing a"**

Using one example throughout the walkthrough provides consistency and makes it possible to observe how the hidden representations evolve as they pass through the network.

Each intermediate tensor can be traced back to the original input, concepts introduced early remain relevant throughout the walkthrough, and every transformation contributes to a coherent picture of how GPT-2 arrives at its prediction.

The result is a complete mental map of the Transformer's forward pass rather than a collection of isolated concepts.

---

## Why focus on inference?

A key advantage of focusing on inference is that it allows us to study a real, production-quality language model.

Training GPT-2 from scratch requires enormous datasets, specialized hardware, and weeks of computation. Consequently, training-focused tutorials necessarily simplify the problem by using tiny datasets, miniature models, and abbreviated training procedures. While these examples are valuable for understanding optimization and gradient descent, the resulting models bear little resemblance to GPT-2 and cannot be directly compared with the official implementation.

Inference offers something different: exact reproducibility.

Because GPT-2 Small's pretrained parameters are publicly available, we can independently implement every layer using only the published weights and the mathematical definitions of each operation. Every hidden state, attention matrix, activation, and output can then be compared against the official implementation, confirming that the implementation faithfully reproduces the behavior of GPT-2.

Rather than constructing a simplified Transformer for teaching purposes, this repository reconstructs the computations performed by GPT-2 itself.

---

## Interactive notebook

The entire walkthrough is provided as an interactive **Google Colab notebook**.

Readers can execute every computation, inspect intermediate tensors, experiment with the code, and immediately compare their results against Hugging Face's implementation.

Since the walkthrough focuses exclusively on inference, it runs comfortably on the free Google Colab runtime.

---

## Intended audience

This repository is intended for:

- Data scientists
- Machine learning practitioners
- AI engineers
- Researchers learning Transformer models
- Anyone who wants to understand GPT-2 beyond architectural diagrams

Readers should be comfortable with:

- Linear algebra
- Vectors and matrices
- Basic probability
- Python

Prior experience with deep learning is helpful but not required.

---

## Repository philosophy

This project is built around five principles:

- **One running example** from beginning to end.
- **Direct implementation of every layer** from its mathematical definition.
- **No skipped mathematics.**
- **Verification against the official implementation at every stage.**
- **Mathematical rigor combined with practical intuition.**

The goal is not simply to explain the Transformer architecture, but to develop a complete understanding of how a modern language model transforms an input sequence into its next-token prediction by independently reconstructing every computation performed during inference.
