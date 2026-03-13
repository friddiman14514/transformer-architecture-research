# Architectural Components of the Transformer

## Self‑Attention

Self‑attention computes a weighted sum of values (V) where the weights are derived from the similarity between queries (Q) and keys (K). The core operation is:
```
Attention(Q, K, V) = softmax((QKᵀ) / √d_k) V
```
where *d_k* is the dimensionality of the keys. This mechanism enables each token to incorporate information from all other tokens.

## Multi‑Head Attention

To capture different types of relationships, the model splits the embedding dimension into *h* heads, each with its own learned linear projections of Q, K, V. The heads are concatenated and linearly transformed:
```
MultiHead(Q, K, V) = Concat(head₁,…,head_h)W⁰
```
This allows the model to attend to information from multiple representation subspaces simultaneously.

## Positional Encoding

Since the architecture lacks recurrence, positional encodings are added to token embeddings to inject order information. The original paper uses sinusoidal functions:
```
PE_{(pos,2i)}   = sin(pos / 10000^{2i/d_model})
PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_model})
```
These encodings enable the model to reason about relative and absolute positions.

## Encoder Stack

Each encoder layer consists of:
1. **Multi‑Head Self‑Attention** (with residual connection and layer norm)
2. **Position‑wise Feed‑Forward Network** (two linear layers with ReLU, plus residual + layer norm)

Six identical encoder layers are stacked, allowing the model to build hierarchical representations.

## Decoder Stack

Each decoder layer includes:
1. **Masked Multi‑Head Self‑Attention** (prevents attending to future tokens)
2. **Multi‑Head Encoder‑Decoder Attention** (queries the encoder output)
3. **Position‑wise Feed‑Forward Network**

Again, residual connections and layer normalization are applied after each sub‑layer.

## Additional Innovations
- **Layer Normalization** after each sub‑layer stabilizes training.
- **Residual Connections** help gradients flow through deep stacks.
- **Dropout** is applied to attention weights and feed‑forward outputs for regularization.
