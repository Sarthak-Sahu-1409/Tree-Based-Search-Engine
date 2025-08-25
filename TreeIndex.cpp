// TreeIndex.cpp: Implements two spatial index structures used for approximate/accelerated NN search.
// OOP highlights: two concrete subclasses (KDTreeIndex, RPTreeIndex) of the abstract interface in
// TreeIndex.h; both manage lifetime of a Node-based tree, expose uniform Add/Remove/Search APIs,
// and use the Singleton pattern (GetInstance) for global access with configurable leaf size.

#include <bits/stdc++.h>
#include <random>
#include "TreeIndex.h"
using namespace std;

// Process-wide random number generator for utilities that need randomness.
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dis(-1.0, 1.0);

// --- KD Tree implementation ---
// KDTreeIndex partitions space along axis-aligned hyperplanes. The split rule chooses the
// dimension with maximum spread and splits at the median. Leaves hold up to M points.

void KDTreeIndex::AddData(const vector<DataVector> &newDataset)
{
    // Append new data and rebuild the tree to maintain search guarantees.
    dataset.insert(dataset.end(), newDataset.begin(), newDataset.end());
    
    if (root)
    {
        delete root;
        root = nullptr;
    }
    MakeTree();
}

void KDTreeIndex::RemoveData(const vector<DataVector> &dataToRemove)
{
    // Linear remove-by-value (exact equality) followed by rebuild. For large datasets,
    // a lazy delete or rebuild-once batch strategy would be preferable.
    for(const DataVector &data : dataToRemove)
    {
        for(int i = 0; i < dataset.size(); i++)
        {
            if(dataset[i] == data){
                dataset.erase(dataset.begin() + i);
                break;
            }
        }
    }
    
    if (root)
    {
        delete root;
        root = nullptr;
    }
    MakeTree();
}

void KDTreeIndex::Search(const DataVector &testVector, int k)
{
    if (dataset.empty())
    {
        cout << "Error: empty dataset" << endl;
        return;
    }

    vector<int> nearestIndices;
    vector<double> distances;

    // Best-first search over the built KD tree; collects up to k nearest indices and distances.
    searchTree(root, testVector, k, nearestIndices, distances);

    cout << "Nearest neighbors:" << endl;
    for (int i = 0; i < nearestIndices.size(); ++i)
    {
        cout << "Neighbor " << i + 1 << ": Index = " << nearestIndices[i] << endl;
        cout<<" Distance = " << distances[i] << endl;
        // cout<<"Vector: ";
        // dataset[nearestIndices[i]].print();
        cout<<endl;
    }
}

// Iterative tree traversal: descend to a candidate leaf, then backtrack while checking whether
// crossing the split hyperplane could reveal closer points. Maintains a simple k-best buffer.
void KDTreeIndex::searchTree(Node* node, const DataVector& testVector, int k, vector<int>& nearestIndices, vector<double>& distances) {
    if (node == nullptr) {
        return;
    }
    
    stack<Node*> path;
    Node* current = node;

    // Traverse down to a leaf guided by split comparisons
    while (current != nullptr) {
        path.push(current);
        if (current->vectorIndices.empty()) {
            current = nullptr;
        } else if (testVector.getComponent(current->splitDim) <= dataset[current->vectorIndices[0]].getComponent(current->splitDim)) {
            current = current->leftChild;
        } else {
            current = current->rightChild;
        }
    }

    double bestDistance = numeric_limits<double>::infinity();
    Node* bestNode = nullptr;

    // Track which indices we've already inserted to avoid duplicates
    unordered_set<int> addedIndices;

    // Backtrack: evaluate candidate points and decide whether to explore the opposite branch
    while (!path.empty()) {
        current = path.top();
        path.pop();

        for (int index : current->vectorIndices) {
            if (index < dataset.size()) { // Ensure index is within bounds
                double distance = testVector.dist(dataset[index]);
                if (nearestIndices.size() < k) {
                    if (addedIndices.find(index) == addedIndices.end()) {
                        nearestIndices.push_back(index);
                        distances.push_back(distance);
                        addedIndices.insert(index);
                    }
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestNode = current;
                    }
                }
                else {
                     if (distance < bestDistance) {
                        if (addedIndices.find(index) == addedIndices.end()) {
                            addedIndices.erase(nearestIndices.back());
                            nearestIndices.pop_back();
                            distances.pop_back();
                            nearestIndices.push_back(index);
                            distances.push_back(distance);
                            addedIndices.insert(index);
                            bestDistance = distance;
                            bestNode = current;
                        }
                    }
                }
            }
        }

        // Distance to the splitting hyperplane at this node along the split dimension
        if (!current->vectorIndices.empty()) {
            double splitDistance = abs(testVector.getComponent(current->splitDim) - dataset[current->vectorIndices[0]].getComponent(current->splitDim));

            // If the distance to the splitting hyperplane is less than the best distance, search the other side of the tree
            if (splitDistance < bestDistance) {
                if (testVector.getComponent(current->splitDim) <= dataset[current->vectorIndices[0]].getComponent(current->splitDim)) {
                    current = current->rightChild;
                } else {
                    current = current->leftChild;
                }
                while (current != nullptr) {
                    path.push(current);
                    if (current->vectorIndices.empty()) {
                        current = nullptr;
                    } else if (testVector.getComponent(current->splitDim) <= dataset[current->vectorIndices[0]].getComponent(current->splitDim)) {
                        current = current->leftChild;
                    } else {
                        current = current->rightChild;
                    }
                }
            }
        }
    }

    // Finalize results sorted by ascending distance
    vector<pair<int, double>> nearestPairs;
    for (int i = 0; i < nearestIndices.size(); ++i) {
        nearestPairs.push_back(make_pair(nearestIndices[i], distances[i]));
    }
    sort(nearestPairs.begin(), nearestPairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
        return a.second < b.second;
    });

    // Update the nearest indices and distances
    for (int i = 0; i < nearestPairs.size(); ++i) {
        nearestIndices[i] = nearestPairs[i].first;
        distances[i] = nearestPairs[i].second;
    }
}

void KDTreeIndex::buildTree(Node *&node, const vector<int> &indices)
{
    if (indices.empty())
    {
        return;
    }

    // Leaf termination: fewer than M points
    if (indices.size() < M)
    {
        node->isLeaf = true;
        return;
    }

    int splitDim;
    auto Rule = ChooseRule(indices, splitDim);
    
    node->vectorIndices = indices;
    
    // Stable partition by split rule into left/right children
    vector<int> leftIndices, rightIndices;
    for (int index : indices)
    {
        if (Rule(dataset[index]))
        {
            leftIndices.push_back(index);
        }
        else
        {
            rightIndices.push_back(index);
        }
    }
    
     if (leftIndices.empty() || rightIndices.empty())
    {
        // Degenerate split; stop splitting further
        return;
    }

    // Recursively build children
    node->splitDim = splitDim;
    node->leftChild = new Node();
    node->rightChild = new Node();
    buildTree(node->leftChild, leftIndices);
    buildTree(node->rightChild, rightIndices);
}

// Choose axis-aligned split: pick dimension of max spread; threshold at median.
function<bool(const DataVector &)> KDTreeIndex::ChooseRule(const vector<int> &indices, int &splitDim) {
    if (indices.empty()) {
        throw invalid_argument("Empty subset");
    }
    
    int numDims = dataset[indices[0]].getDimension();
    vector<double> maxVals(numDims, numeric_limits<double>::lowest());
    vector<double> minVals(numDims, numeric_limits<double>::max());

    // Compute per-dimension min/max on the subset
    for (int index : indices) {
        for (int i = 0; i < numDims; ++i) {
            double val = dataset[index].getComponent(i);
            if (val > maxVals[i]) {
                maxVals[i] = val;
            }
            if (val < minVals[i]) {
                minVals[i] = val;
            }
        }
    }
    
    double maxSpread = numeric_limits<double>::lowest();

    // Choose dimension with maximum spread
    for (int i = 0; i < numDims; ++i) {
        double spread = maxVals[i] - minVals[i];
        if (spread > maxSpread) {
            maxSpread = spread;
            splitDim = i;
        }
    }

    // Compute median threshold along the chosen dimension
    vector<double> dimVals;
    for (int index : indices) {
        dimVals.push_back(dataset[index].getComponent(splitDim));
    }
    sort(dimVals.begin(), dimVals.end());
    double median = dimVals[dimVals.size() / 2];

    // Split rule: goes left if value <= median
    return [splitDim, median](const DataVector& vec) {
        return vec.getComponent(splitDim) <= median;
    };
}

void KDTreeIndex::MakeTree()
{
    if (root)
    {
        delete root;
        root = nullptr;
    }

    if (!dataset.empty())
    {
        vector<int> indices(dataset.size());
        iota(indices.begin(), indices.end(), 0); // 0..n-1 indices of dataset
        root = new Node();
        buildTree(root, indices);
    }
}

// Singleton accessor (OOP: Singleton pattern)
KDTreeIndex &KDTreeIndex::GetInstance(int leafSize)
{
    static KDTreeIndex instance(leafSize);
    return instance;
}

void KDTreeIndex::printNodeIndices(Node *node, int depth)
{
    if (node == nullptr)
    {
        if (depth == 0)
        {
            cout << "Root node is nullptr" << endl;
        }
        else
        {
            cout << "Node at depth " << depth << " is nullptr" << endl;
        }
        return;
    }
    
    printNodeIndices(node->leftChild, depth + 1);

    if (node->vectorIndices.size())
    {
        cout << "Indices in this node: ";
        for (int index : node->vectorIndices)
        {
            cout << index << " ";
        }
        cout << endl;
    }

    printNodeIndices(node->rightChild, depth + 1);
}

double uniform_random(double min, double max)
{
    return min + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (max - min)));
}

// --- RP Tree implementation ---
// RPTreeIndex uses random projections and median thresholding with an additional random shift (delta)
// to reduce worst-case degeneration on structured data. This often balances trees better for high-dim data.

void RPTreeIndex::AddData(const vector<DataVector> &newDataset)
{
    // Append and rebuild, mirroring KDTreeIndex semantics.
    dataset.insert(dataset.end(), newDataset.begin(), newDataset.end());
    
    if (root)
    {
        delete root;
        root = nullptr;
    }
    MakeTree();
}

void RPTreeIndex::RemoveData(const vector<DataVector> &dataToRemove)
{
    // Linear remove followed by rebuild (same caveat as KDTree).
    for(const DataVector &data : dataToRemove)
    {
        for(int i = 0; i < dataset.size(); i++)
        {
            if(dataset[i] == data){
                dataset.erase(dataset.begin() + i);
                break;
            }
        }
    }
    
    if (root)
    {
        delete root;
        root = nullptr;
    }
    MakeTree();
}

void RPTreeIndex::Search(const DataVector &testVector, int k)
{
    if (dataset.empty())
    {
        cout << "Error: empty dataset" << endl;
        return;
    }

    vector<int> nearestIndices;
    vector<double> distances;

    searchTree(root, testVector, k, nearestIndices, distances);

    cout << "Nearest neighbors:" << endl;
    for (int i = 0; i < nearestIndices.size(); ++i)
    {
        cout << "Neighbor " << i + 1 << ": Index = " << nearestIndices[i] <<endl;
        //  ", Distance = " << distances[i] << endl;
        // cout<<"Vector: ";
        // dataset[nearestIndices[i]].print();
        cout<<endl;
    }
}

// Same search strategy as KDTreeIndex; only the splitting rule differs.
void RPTreeIndex::searchTree(Node* node, const DataVector& testVector, int k, vector<int>& nearestIndices, vector<double>& distances) {
    if (node == nullptr) {
        return;
    }
    
    stack<Node*> path;
    Node* current = node;

    // Traverse down to a leaf guided by split comparisons
    while (current != nullptr) {
        path.push(current);
        if (current->vectorIndices.empty()) {
            current = nullptr;
        }
        else if (testVector.getComponent(current->splitDim) <= dataset[current->vectorIndices[0]].getComponent(current->splitDim)) {
            current = current->leftChild;
        } else {
            current = current->rightChild;
        }
    }

    double bestDistance = numeric_limits<double>::infinity();
    Node* bestNode = nullptr;

    // Track which indices we've already inserted to avoid duplicates
    unordered_set<int> addedIndices;

    // Backtrack: evaluate candidate points and decide whether to explore the opposite branch
    while (!path.empty()) {
        current = path.top();
        path.pop();

        for (int index : current->vectorIndices) {
            if (index < dataset.size()) { // Ensure index is within bounds
                double distance = testVector.dist(dataset[index]);
                if (nearestIndices.size() < k) {
                    if (addedIndices.find(index) == addedIndices.end()) {
                        nearestIndices.push_back(index);
                        distances.push_back(distance);
                        addedIndices.insert(index);
                    }
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestNode = current;
                    }
                } else {
                     if (distance < bestDistance) {
                        if (addedIndices.find(index) == addedIndices.end()) {
                            addedIndices.erase(nearestIndices.back());
                            nearestIndices.pop_back();
                            distances.pop_back();
                            nearestIndices.push_back(index);
                            distances.push_back(distance);
                            addedIndices.insert(index);
                            bestDistance = distance;
                            bestNode = current;
                        }
                    }
                }
            }
        }

        // Distance to the splitting hyperplane at this node along the split dimension
        if (!current->vectorIndices.empty()) {
            double splitDistance = abs(testVector.getComponent(current->splitDim) - dataset[current->vectorIndices[0]].getComponent(current->splitDim));

            // If the distance to the splitting hyperplane is less than the best distance, search the other side of the tree
            if (splitDistance < bestDistance) {
                if (testVector.getComponent(current->splitDim) <= dataset[current->vectorIndices[0]].getComponent(current->splitDim)) {
                    current = current->rightChild;
                } else {
                    current = current->leftChild;
                }
                while (current != nullptr) {
                    path.push(current);
                    if (current->vectorIndices.empty()) {
                        current = nullptr;
                    } else if (testVector.getComponent(current->splitDim) <= dataset[current->vectorIndices[0]].getComponent(current->splitDim)) {
                        current = current->leftChild;
                    } else {
                        current = current->rightChild;
                    }
                }
            }
        }
    }

    // Finalize results sorted by ascending distance
    vector<pair<int, double>> nearestPairs;
    for (int i = 0; i < nearestIndices.size(); ++i) {
        nearestPairs.push_back(make_pair(nearestIndices[i], distances[i]));
    }
    sort(nearestPairs.begin(), nearestPairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
        return a.second < b.second;
    });

    // Update the nearest indices and distances
    for (int i = 0; i < nearestPairs.size(); ++i) {
        nearestIndices[i] = nearestPairs[i].first;
        distances[i] = nearestPairs[i].second;
    }
}

void RPTreeIndex::buildTree(Node *&node, const vector<int> &indices)
{
    if (indices.empty())
    {
        return;
    }

    // Leaf termination: fewer than M points
    if (indices.size() < M)
    {
        node->isLeaf = true;
        return;
    }

    int splitDim;
    auto Rule = ChooseRule(indices, splitDim);
    
    node->vectorIndices = indices;
    // cout<<indices.size()<<endl;

    // Partition by split rule into left/right children
    vector<int> leftIndices, rightIndices;
    for (int index : indices)
    {
        if (Rule(dataset[index]))
        {
            leftIndices.push_back(index);
        }
        else
        {
            rightIndices.push_back(index);
        }
    }
    // cout<<leftIndices.size()<<endl;
     if (leftIndices.empty() || rightIndices.empty())
    {
        // Degenerate split; stop splitting further
        return;
    }

    // Recursively build children
    node->splitDim = splitDim;
    node->leftChild = new Node();
    node->rightChild = new Node();
    buildTree(node->leftChild, leftIndices);
    buildTree(node->rightChild, rightIndices);
}

// Choose random projection split: random unit vector v, random shift delta, threshold at median dot(v,·)+delta.
function<bool(const DataVector &)> RPTreeIndex::ChooseRule(const vector<int> &indices, int &splitDim) {
     if (indices.empty()) {
        throw invalid_argument("Empty subset");
    }
    
    int numDims = dataset[indices[0]].getDimension();

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-1.0, 1.0);

    // Random unit direction v in R^d
    DataVector v(numDims);
    for (int i = 0; i < numDims; ++i) {
        v.setComponent(i, dis(gen));
    }

    // Normalize to get a unit vector
    v.normalize();
    // v.print();

    // Heuristic: pick a far point y from an arbitrary x to scale delta
    double maxDistance = -1.0;
    DataVector x = dataset[indices[0]];
    DataVector y(numDims);
    for (int index : indices) {
        double distance = x.dist(dataset[index]);
        if (distance > maxDistance) {
            maxDistance = distance;
            y = dataset[index];
        }
    }

    // Random shift improves balance and robustness on clustered data
    double delta = dis(gen) * 6 * sqrt(x.dist(y)) / sqrt(numDims);

    // Median along the projection defines the threshold
    double medianDotProduct = 0.0;
    vector<double> dotProducts;
    for (int index : indices) {
        dotProducts.push_back(dataset[index].dot(v));
    }
    sort(dotProducts.begin(), dotProducts.end());
    int n = dotProducts.size();
    if (n % 2 == 0) {
        medianDotProduct = (dotProducts[n / 2 - 1] + dotProducts[n / 2]) / 2.0;
    } else {
        medianDotProduct = dotProducts[n / 2];
    }
    
    // Split rule with random shift
    return [v, medianDotProduct, delta](const DataVector& vec) {
        return vec.dot(v) <= (medianDotProduct + delta);
    };
    
}

void RPTreeIndex::MakeTree()
{
    if (root)
    {
        delete root;
        root = nullptr;
    }

    if (!dataset.empty())
    {
        vector<int> indices(dataset.size());
        iota(indices.begin(), indices.end(), 0); // 0..n-1 indices of dataset
        root = new Node();
        buildTree(root, indices);
    }
}

// Singleton accessor (OOP: Singleton pattern)
RPTreeIndex &RPTreeIndex::GetInstance(int leafSize)
{
    static RPTreeIndex instance(leafSize);
    return instance;
}

void RPTreeIndex::printNodeIndices(Node *node, int depth)
{
    if (node == nullptr)
    {
        if (depth == 0)
        {
            cout << "Root node is nullptr" << endl;
        }
        else
        {
            cout << "Node at depth " << depth << " is nullptr" << endl;
        }
        return;
    }
    
    printNodeIndices(node->leftChild, depth + 1);

    if (node->vectorIndices.size())
    {
        cout << "Indices in this node: ";
        for (int index : node->vectorIndices)
        {
            cout << index << " ";
        }
        cout << endl;
    }

    printNodeIndices(node->rightChild, depth + 1);
}
