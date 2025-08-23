# K-Nearest Neighbors (KNN) Implementation with Tree-Based Search

A high-performance C++ implementation of K-Nearest Neighbors algorithm featuring both KD-Tree and Random Projection Tree (RP-Tree) indexing structures for efficient similarity search in high-dimensional spaces.

## 🚀 Features

- **Dual Tree Indexing**: Implements both KD-Tree and RP-Tree for optimal performance across different data distributions
- **High-Dimensional Support**: Efficiently handles high-dimensional vector data (e.g., image features, embeddings)
- **Dynamic Operations**: Support for adding and removing data points from existing trees
- **CSV Dataset Support**: Easy integration with CSV datasets and custom data formats
- **Performance Monitoring**: Built-in timing and performance measurement capabilities

## 🏗️ Architecture

### Core Classes

- **`DataVector`**: High-performance vector class with mathematical operations
- **`VectorDataset`**: Dataset management and I/O operations
- **`TreeIndex`**: Abstract base class for tree implementations
- **`KDTreeIndex`**: KD-Tree implementation for spatial partitioning
- **`RPTreeIndex`**: Random Projection Tree for high-dimensional data

### Tree Structures

#### KD-Tree
- **Spatial Partitioning**: Divides space along coordinate axes
- **Median-based Splitting**: Optimizes tree balance and search performance

#### RP-Tree
- **Random Projections**: Uses random hyperplanes for splitting
- **High-Dimensional Optimization**: Better performance for very high-dimensional data

## 📁 Project Structure

```
KNN-main/
├── TreeIndex.cpp          # Main tree implementations
├── TreeIndex.h            # Tree class definitions
├── DataVector.cpp         # Vector operations implementation
├── DataVector.h           # Vector class definition
├── VectorDataset.cpp      # Dataset management
├── VectorDataset.h        # Dataset class definition
├── nearestneighbor.cpp    # Basic KNN implementation
├── run.cpp                # Main program with tree selection
├── Makefile               # Build configuration
├── download_dataset.py    # Dataset conversion script
├── fmnist-test.csv       # Fashion MNIST test dataset
├── testing.csv            # Sample test dataset
└── test-vector.csv       # Sample test vector
```

## 🔧 Building the Project

### Option 1: Using Makefile (Recommended)

```bash
# Build the project
make

# Clean build artifacts
make clean

# Build and run in one command
make all
```

### Option 2: Manual Compilation

```bash
# Compile with optimizations
g++ -Wall -O2 -o TreeIndex.out run.cpp DataVector.cpp TreeIndex.cpp

# Run the program
./TreeIndex.out
```

## 🚀 Usage

### Basic KNN Search

```bash
# Run the basic nearest neighbor program
./nearest_neighbor_program.exe
```

### Tree-Based Search

```bash
# Run the advanced tree-based search
./TreeIndex.out
```

### Interactive Configuration

The program will prompt you for:
1. **Leaf Size**: Number of vectors per leaf node (default: 200)
2. **Tree Type**: Choose between KD-Tree (1) or RP-Tree (2)
3. **K Value**: Number of nearest neighbors to find
4. **Test Cases**: Number of test vectors to process

## 📊 Dataset Format

### Input Format
- **CSV Files**: Comma-separated values
- **Vector Format**: Each row represents a vector, each column a dimension
- **Header**: Optional (will be skipped if present)

### Example Dataset
```csv
1.2,3.4,5.6,7.8
2.1,4.3,6.5,8.7
3.0,5.2,7.4,9.6
```

## 🎯 Performance Characteristics

### KD-Tree
- **Best for**: Low to medium dimensional data (< 20 dimensions)
- **Search Time**: O(log n) average case
- **Memory**: Moderate overhead
- **Construction**: O(n log n)

### RP-Tree
- **Best for**: High dimensional data (> 20 dimensions)
- **Search Time**: O(log n) with high probability
- **Memory**: Lower overhead than KD-Tree
- **Construction**: O(n log n)

## 🔍 Example Use Cases

- **Image Similarity Search**: Find similar images using feature vectors
- **Recommendation Systems**: Find similar users or items
- **Anomaly Detection**: Identify outliers in high-dimensional data
- **Clustering**: Group similar data points
- **Machine Learning**: Preprocessing and feature matching

## 📈 Performance Tips

1. **Leaf Size Tuning**: 
   - Smaller leaves: Faster search, more memory
   - Larger leaves: Slower search, less memory
   - Default (200) works well for most cases

2. **Tree Selection**:
   - Use KD-Tree for low-dimensional data (< 20D)
   - Use RP-Tree for high-dimensional data (> 20D)

3. **Dataset Preparation**:
   - Normalize vectors for consistent distance calculations
   - Remove duplicate vectors to improve tree efficiency

## 🐛 Troubleshooting

### Common Issues

1. **Compilation Errors**:
   - Ensure GCC/G++ is installed and in PATH
   - Check C++11 support: `g++ --version`

2. **Memory Issues**:
   - Reduce leaf size for large datasets
   - Process data in smaller batches

3. **Performance Issues**:
   - Use appropriate tree type for data dimensionality
   - Optimize leaf size based on dataset characteristics

## 🤝 Contributing

This project is designed for educational and research purposes. Feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Use in your own projects

## 📚 References

- **KD-Tree**: Bentley, J. L. (1975). Multidimensional binary search trees used for associative searching
- **RP-Tree**: Dasgupta, S., & Freund, Y. (2008). Random projection trees and low dimensional manifolds
- **KNN Algorithm**: Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Built with ❤️ in C++** | **Performance-focused KNN implementation**
