# OriginScale
(An attempt to make the fastest geometrically initialisable clustering algorithm) 
A deterministic, parameter-free centroid initializer for K-means clustering.

OriginScale selects initial centroids by sorting all points by their L2 distance from the coordinate origin and taking the `k` closest points. One sort. No randomness. No hyperparameters.

```python
from originscale import OriginScale

model = OriginScale(n_clusters=3)
model.fit(X)          # X must be StandardScaler-normalized
labels = model.labels_
```

---

## How it works

```python
def _init_centroids(self, X):
    idx = np.argsort(np.linalg.norm(X, axis=1))
    return X[idx[:self.n_clusters]].copy()
```

Initialization cost: **O(n log n)** for the sort, independent of k.  
Standard Lloyd iterations follow. Total space: **O(n)**.

> **Requirement:** Input must be zero-centered. Apply `StandardScaler` before fitting.

---

## Benchmark results

All experiments run on StandardScaler-normalized data. All centroid methods use `n_init=1` for fair comparison.

### Quality — 7 datasets, n=150–1000 (Silhouette score, higher is better)

| Dataset | OriginScale | KMeans++ (n=1) | KMeans rand (n=1) | MiniBatch |
|---|---|---|---|---|
| Moons | 0.4904 | 0.4905 | 0.4904 | 0.4903 |
| Blobs | **0.7983** | **0.7983** | **0.7983** | **0.7983** |
| Circles | 0.2933 | 0.2961 | 0.2917 | 0.2923 |
| Anisotropic | **0.8141** | **0.8141** | **0.8141** | **0.8141** |
| Iris | 0.4565 | 0.4799 | 0.4630 | 0.4838 |
| Wine | **0.2849** | **0.2849** | 0.2807 | 0.2106 |
| Breast Cancer | 0.3434 | 0.3447 | 0.3447 | 0.3427 |

Matches KMeans++ on 5 of 7 datasets. Maximum gap: 0.023 (Iris).

### Reproducibility — 20 unseeded runs (Silhouette std, lower is better)

| Dataset | OriginScale | KMeans++ (n=1) | KMeans rand (n=1) |
|---|---|---|---|
| Blobs | **0.000000** | 0.000000 | 0.120940 |
| Iris | **0.000000** | 0.011025 | 0.007629 |
| Wine | **0.000000** | 0.001906 | 0.001669 |

OriginScale produces identical results on every run by construction. KMeans with random initialization ranged from silhouette 0.3941 to 0.7983 on the blobs dataset across 20 runs.

### Initialization time vs k — n=200,000, d=20

| k | OriginScale | KMeans++ | MiniBatch init | PCA-based |
|---|---|---|---|---|
| 3 | **25ms** | 174ms | 79ms | 54ms |
| 10 | **27ms** | 280ms | 95ms | 91ms |
| 25 | **27ms** | 499ms | 104ms | 117ms |
| 50 | **26ms** | 823ms | 139ms | 139ms |
| 100 | **26ms** | 1,641ms | 204ms | 153ms |
| 200 | **26ms** | 3,533ms | 373ms | 174ms |

OriginScale initialization time is **constant with respect to k** because it requires only a sort. KMeans++ scales as O(n·k).

### Large-scale convergence — n=200,000, d=20

| k | Init | t_init | Iterations | Final inertia |
|---|---|---|---|---|
| 10 | OriginScale | 30ms | 59 | 2,855,526 |
| 10 | KMeans++ | 829ms | 41 | 2,836,958 |
| 50 | OriginScale | 26ms | 39 | 1,487,112 |
| 50 | KMeans++ | 1,419ms | 9 | 1,291,883 |
| 100 | OriginScale | 32ms | 82 | 735,969 |
| 100 | KMeans++ | 2,233ms | 58 | 299,409 |

At k=100, OriginScale initializes **69× faster** than KMeans++ but converges to a higher inertia. This is the core trade-off: OriginScale is faster to initialize but produces a weaker starting point at large k, requiring more Lloyd iterations and yielding a higher final objective.

### Memory usage — n=200,000, k=50

| Initializer | Peak memory |
|---|---|
| Random | 1,574 KB |
| MiniBatch init | 4,583 KB |
| PCA-based | 14,073 KB |
| **OriginScale** | **34,376 KB** |
| KMeans++ | 62,567 KB |

---

## Complexity

| Initializer | Time | Space | Deterministic | Param-free |
|---|---|---|---|---|
| OriginScale | O(n log n) | O(n) | Yes | Yes |
| KMeans++ | O(n·k) | O(n) | No | No (seed) |
| Random | O(n) | O(k) | No | No (seed) |
| MiniBatch init | O(b·k·t) | O(b) | No | No (seed, batch) |
| PCA-based | O(n·d·log k) | O(n·c) | Yes | Yes |

---

## When to use OriginScale

**Good fit:**
- You need reproducible cluster assignments across retraining runs
- You are running many clustering jobs and want to eliminate seed management
- k is large (k ≥ 50) and initialization cost is a bottleneck
- Memory is constrained relative to KMeans++

**Not a good fit:**
- Minimizing final inertia is the primary objective and you can afford KMeans++ with multiple restarts
- Data cannot be zero-centered (StandardScaler not applicable)

---

## Off-center sensitivity

OriginScale assumes zero-centered input. On un-normalized data with a translation offset applied post-scaling, quality degrades:

| Dataset | Offset 0 | Offset 2 | Offset 5 | KMeans++ (any offset) |
|---|---|---|---|---|
| Blobs | 0.7983 | 0.5949 | 0.3946 | 0.7983 |
| Anisotropic | 0.8141 | 0.6201 | 0.6201 | 0.8141 |
| Moons | 0.4904 | 0.4905 | 0.4905 | 0.4905 |
| Iris | 0.4565 | 0.4565 | 0.4565 | 0.4799 |

**Apply StandardScaler before fitting.** All benchmark results above used StandardScaler.

---

## Requirements

```
python >= 3.8
numpy >= 1.19
scikit-learn >= 0.24   # for metrics and preprocessing only
```

---

## Citation

```bibtex
@software{girish2024originscale,
  author = {Girish, Aditya},
  title  = {OriginScale: A Deterministic Centroid Initializer for K-means},
  year   = {2024},
  url    = {https://github.com/adityagirishh/OriginScale-a-novel-initialisation}
}
```
