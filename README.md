# Transformer Architecture Research

## Introduction

Transformers are a class of deep learning models introduced in 2017 by Vaswani et al. in the seminal paper *"Attention Is All You Need"*. Unlike previous sequence models that relied on recurrence or convolutions, Transformers are built entirely on attention mechanisms, enabling highly parallelizable computation and superior performance on a wide range of tasks.

Key components include:
- **Self‑Attention**: Allows each token to attend to every other token in the sequence, capturing long‑range dependencies.
- **Multi‑Head Attention**: Splits attention into multiple sub‑spaces to learn different relational patterns.
- **Positional Encoding**: Injects order information since the model lacks recurrence.
- **Encoder‑Decoder Structure**: Stacks of identical encoder and decoder layers (typically 6 each) that process inputs and generate outputs.
- **Layer Normalization & Residual Connections**: Stabilize training and enable deeper stacks.

## Key Innovations

The *Attention Is All You Need* paper introduced several architectural innovations that set the foundation for modern NLP models:
1. **Scaled Dot‑Product Attention** – computes attention scores as the dot product of queries and keys, scaled by the square root of dimensionality.
2. **Multi‑Head Attention** – runs several attention mechanisms in parallel, concatenating their outputs.
3. **Position‑wise Feed‑Forward Networks** – simple fully‑connected layers applied independently to each position.
4. **Positional Encodings** – sinusoidal functions added to token embeddings to convey positional information.
5. **Fully Parallelizable Architecture** – removal of recurrence enables training on GPUs/TPUs with significantly reduced time.

These innovations have been adopted and extended in models such as BERT, GPT, T5, Vision Transformers, and many others.

## Historical Context

- **2017** – *Attention Is All You Need* introduces the Transformer.
- **2018** – BERT (Devlin et al.) popularizes pre‑training using Transformer encoders.
- **2019** – GPT‑2 demonstrates large‑scale language generation with decoder‑only Transformers.
- **2020** – Vision Transformers (ViT) adapt the architecture to computer vision.
- **2021‑2023** – Numerous variants (e.g., Efficient Transformers, Longformer, Performer) address scaling and efficiency.
- **2024‑present** – Ongoing research explores sparse attention, multimodal Transformers, and integration with retrieval‑augmented models.

## References
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. https://arxiv.org/abs/1706.03762
- Wikipedia contributors. *Transformer (machine learning architecture)*. https://en.wikipedia.org/wiki/Transformer_(machine_learning_architecture)
