# K-Nearest Neighbors (KNN) with KD-Tree and RP-Tree

A C++ implementation of K-Nearest Neighbors featuring two indexing structures: KD-Tree and Random Projection Tree (RP-Tree). This repository includes a simple KNN driver as well as an interactive demo to build/search trees over CSV datasets.

## Highlights

- Dual indexing: KD-Tree (axis-aligned splits) and RP-Tree (random projection splits)
- Value-type math utilities (`DataVector`) and dataset loader (`VectorDataset`)
- Add/remove data and rebuild indices
- CSV input for quick experiments
- Simple CLI tools for both plain KNN and tree-based search

## Repository Layout

```
KNN-main/
├── DataVector.cpp / .h        # Numeric vector type, math ops, CSV reader (per-line to vectors)
├── VectorDataset.cpp / .h     # Container and CSV ingestion for datasets
├── TreeIndex.cpp / .h         # KDTreeIndex and RPTreeIndex implementations + interface
├── nearestneighbor.cpp        # Baseline KNN (brute-force with sorting)
├── run.cpp                    # Interactive tree demo (choose KDTree/RPTree)
├── Makefile                   # Build targets
├── download_dataset.py        # Helper script (optional)
├── testing.csv                # Sample dataset
├── test-vector.csv            # Sample test vectors
├── fmnist-test.csv            # Example larger dataset
└── assignment-*.pdf, scripts  # Course artifacts and helper scripts
```

## Build

### Windows (PowerShell)

Prerequisites: a C++17 compiler (e.g., MSYS2/MinGW-w64 or Visual Studio Build Tools).

```
# Using g++ (MinGW-w64)
g++ -std=gnu++17 -O2 -Wall -o nn.exe nearestneighbor.cpp DataVector.cpp VectorDataset.cpp
g++ -std=gnu++17 -O2 -Wall -o trees.exe run.cpp DataVector.cpp VectorDataset.cpp TreeIndex.cpp
```

### Linux/macOS

```
# With Makefile
make            # builds executables per Makefile
make clean      # removes artifacts

# Or manual
g++ -std=c++17 -O2 -Wall -o nn nearestneighbor.cpp DataVector.cpp VectorDataset.cpp
g++ -std=c++17 -O2 -Wall -o trees run.cpp DataVector.cpp VectorDataset.cpp TreeIndex.cpp
```

## Run

### Baseline KNN (brute force)

```
# Windows
./nearest_neighbor_program.exe

# Or if built manually
./nn.exe
```

Behavior:
- Loads `testing.csv` as the reference dataset and `test-vector.csv` as query vectors.
- Uses Euclidean distance, sorts all distances, and prints top-k neighbors with timing.

### Tree-based demo (KD-Tree or RP-Tree)

```
# Linux/macOS
./TreeIndex.out

# Windows or manual build
./trees.exe
```

Interactive flow:
1. Enter leaf size (controls maximum points in a leaf node).
2. Training set: reads `test1.csv` into memory.
3. Test set: reads `test2.csv`.
4. Choose 1 (KDTree) or 2 (RPTree), provide k and number of test queries.
5. Program builds the chosen index and runs searches, printing nearest neighbors and distances.

## Data Format

- CSV; each line is one vector; values are comma-separated.
- No header required. If present, ensure the first line is compatible (or pre-strip it).
- Example:

```
0.12,0.33,1.50
0.05,-0.23,0.77
```

## Design and OOP Notes

### `DataVector`
- Encapsulates a `vector<double>` with math operations: `+`, `-`, dot `operator*`, `dot()`, `dist()`, `norm()`, `normalize()`.
- Value semantics: copy constructor and assignment operator copy the underlying data.
- CSV line parsing helper in `readDataset(filename, vector<DataVector>&)` for convenience.

### `VectorDataset`
- Thin container owning `vector<DataVector>` plus a CSV reader.
- Value semantics: safe to copy/assign datasets.

### `TreeIndex` (interface), `KDTreeIndex`, `RPTreeIndex`
- Abstract interface defines `AddData`, `RemoveData`, `MakeTree`, and `Search`.
- KDTreeIndex: axis-aligned splits using max-spread dimension and median threshold.
- RPTreeIndex: random projection vector with random shift (delta), then median threshold.
- Both expose a Singleton accessor `GetInstance(leafSize)` to share configuration globally.

### Search Strategy
- Descend to a candidate leaf by comparing against split rules.
- Backtrack with a stack, exploring the opposite branch when its split distance could beat current best.
- Maintain up to k nearest neighbors and distances; sort results at the end.

## Complexity (informal)

- Baseline KNN: O(n·d) distance computations + O(n log n) sort per query.
- KDTree/RPTree: O(log n) average query for well-behaved data; degrades toward O(n) for adversarial/high-D.
- Build: roughly O(n log n) with median-based partitioning.

## Extending the Project

- Distance metrics: add L1/cosine by extending `DataVector` (e.g., `cosineSimilarity`).
- Persistence: serialize trees for reuse instead of rebuilding each run.
- Batch queries: stream multiple queries and report aggregated timings.
- Parallelism: parallelize distance loops or tree build.

## Troubleshooting

- Build errors on Windows: ensure MinGW-w64 is on PATH or use Visual Studio Developer Prompt.
- Runtime errors opening files: verify CSV paths and working directory; files are resolved relative to the executable.
- Index out of bounds during search: ensure test set size is ≥ requested number of test cases.

## Scripts

- `download_dataset.py`: helper to fetch/convert datasets (optional; inspect before use).
- `script1.sh`, `script2.sh`: shell helpers (optional; platform dependent).

---

Built with C++17. Suitable for teaching, experimentation, and as a foundation for more advanced ANN systems.
