# Understanding Why We Need Embeddings in Deep Learning

### 1. Problem with Using Raw IDs

- Tokens are often mapped to **IDs**: e.g., `"我" → 4`, `"爱" → 5`, `"英国" → 7`.
- These IDs are just **labels**, not meaningful numbers.
- If you feed IDs directly into a model, it may wrongly interpret **numerical relations** (e.g., `7 > 4`).
- An ID alone carries almost **no semantic information**.

---

### 2. One-Hot Encoding

- A safer way: represent each token as a **one-hot vector** of size `V` (vocabulary size).
- Example (`V=10`):

  - `"我" (id=4)` → `[0,0,0,0,1,0,0,0,0,0]`

- Pros:

  - Removes false numerical order.

- Cons:

  - **High-dimensional** (size = `V`, e.g., 30k).
  - **Sparse** (mostly zeros).
  - Inefficient and suffers from the **curse of dimensionality**.
  - No semantic closeness — `"中国"` and `"英国"` are just orthogonal vectors.

---

### 3. Embeddings

- Solution: learn a dense, low-dimensional representation.
- Define an **embedding matrix** `E ∈ R^{V × d_model}`:

  - `V` = vocabulary size (number of tokens).
  - `d_model` = embedding dimension (e.g., 32, 64, 128).

- Each token id corresponds to one **row in `E`**.
- Example (`d_model=4`):

  - `"我" (id=4)` → `[0.12, -0.05, 0.33, 0.87]`

---

### 4. Why Embeddings Work Better

- **Dimensionality reduction**: from 30k → 64 dimensions.
- **Dense representation**: every dimension carries information.
- **Trainable**: embeddings are optimized with the model, so:

  - Semantically similar tokens (e.g., `"中国"`, `"英国"`) end up closer in vector space.
  - Dissimilar tokens end up farther apart.

- **Continuous space**: makes it possible for the model to compute similarities (dot product, cosine) and capture patterns.

---

### 5. Summary Table

| Representation | Dimension per Token | Dense/Sparse | Semantic Meaning | Practicality |
| -------------- | ------------------- | ------------ | ---------------- | ------------ |
| **ID**         | 1                   | N/A          | None             | Misleading   |
| **One-hot**    | V (e.g., 30k)       | Sparse       | None             | Inefficient  |
| **Embedding**  | d_model (e.g., 64)  | Dense        | Learned          | Efficient ✅ |

---

👉 **Key takeaway**:
Embeddings act as a **bridge** from discrete token IDs to a **continuous semantic space** that deep learning models can understand and exploit.
