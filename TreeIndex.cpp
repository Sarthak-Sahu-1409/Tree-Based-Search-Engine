// Two tree indexes for faster nearest-neighbor search.
// KDTreeIndex and RPTreeIndex both implement the TreeIndex interface.
// They build a tree of nodes and provide Add/Remove/Search. A simple Singleton is used.

#include <bits/stdc++.h>
#include <random>
#include "TreeIndex.h"
using namespace std;

// Random number generator used where needed.
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dis(-1.0, 1.0);

// --- KD Tree ---
// Splits along coordinate axes. Pick the widest dimension and split at the median.

void KDTreeIndex::AddData(const vector<DataVector> &newDataset)
{
    // Append and rebuild.
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
    // Remove exact matches, then rebuild. (Simple but not the fastest.)
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

    // Traverse the tree and collect up to k nearest.
    searchTree(root, testVector, k, nearestIndices, distances);

    cout << "Nearest neighbors:" << endl;
    for (int i = 0; i < nearestIndices.size(); ++i)
    {
        cout << "Neighbor " << i + 1 << ": Index = " << nearestIndices[i] << endl;
        cout<<" Distance = " << distances[i] << endl;
        cout<<endl;
    }
}

// Go down to a candidate leaf, then backtrack when the other side might be closer.
void KDTreeIndex::searchTree(Node* node, const DataVector& testVector, int k, vector<int>& nearestIndices, vector<double>& distances) {
    if (node == nullptr) {
        return;
    }
    
    stack<Node*> path;
    Node* current = node;

    // Go down by comparing the split value.
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

    // Avoid duplicates.
    unordered_set<int> addedIndices;

    // Backtrack and check if the other side can beat the best distance.
    while (!path.empty()) {
        current = path.top();
        path.pop();

        for (int index : current->vectorIndices) {
            if (index < dataset.size()) {
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

        // Distance to the split hyperplane for this node.
        if (!current->vectorIndices.empty()) {
            double splitDistance = abs(testVector.getComponent(current->splitDim) - dataset[current->vectorIndices[0]].getComponent(current->splitDim));

            // If the other side might be closer, walk down that side as well.
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

    // Sort by distance.
    vector<pair<int, double>> nearestPairs;
    for (int i = 0; i < nearestIndices.size(); ++i) {
        nearestPairs.push_back(make_pair(nearestIndices[i], distances[i]));
    }
    sort(nearestPairs.begin(), nearestPairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
        return a.second < b.second;
    });

    // Write back ordered results.
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

    // Stop if this node is small enough.
    if (indices.size() < M)
    {
        node->isLeaf = true;
        return;
    }

    int splitDim;
    auto Rule = ChooseRule(indices, splitDim);
    
    node->vectorIndices = indices;
    
    // Split into left/right by the rule.
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
        // Give up splitting if it would be empty on one side.
        return;
    }

    // Build children.
    node->splitDim = splitDim;
    node->leftChild = new Node();
    node->rightChild = new Node();
    buildTree(node->leftChild, leftIndices);
    buildTree(node->rightChild, rightIndices);
}

// Choose the axis with the widest spread; split at the median.
function<bool(const DataVector &)> KDTreeIndex::ChooseRule(const vector<int> &indices, int &splitDim) {
    if (indices.empty()) {
        throw invalid_argument("Empty subset");
    }
    
    int numDims = dataset[indices[0]].getDimension();
    vector<double> maxVals(numDims, numeric_limits<double>::lowest());
    vector<double> minVals(numDims, numeric_limits<double>::max());

    // Track min/max per dimension over the subset.
    for (int index : indices) {
        for (int i = 0; i < numDims; ++i) {
            double val = dataset[index].getComponent(i);
            if (val > maxVals[i]) { maxVals[i] = val; }
            if (val < minVals[i]) { minVals[i] = val; }
        }
    }
    
    double maxSpread = numeric_limits<double>::lowest();

    // Pick the largest spread.
    for (int i = 0; i < numDims; ++i) {
        double spread = maxVals[i] - minVals[i];
        if (spread > maxSpread) {
            maxSpread = spread;
            splitDim = i;
        }
    }

    // Median along the chosen axis.
    vector<double> dimVals;
    for (int index : indices) {
        dimVals.push_back(dataset[index].getComponent(splitDim));
    }
    sort(dimVals.begin(), dimVals.end());
    double median = dimVals[dimVals.size() / 2];

    // Go left if value <= median.
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
        iota(indices.begin(), indices.end(), 0); // 0..n-1
        root = new Node();
        buildTree(root, indices);
    }
}

// Singleton access (one global per leaf size parameter).
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

// --- RP Tree ---
// Uses a random direction and a small random shift. Often balances better for high-D data.

void RPTreeIndex::AddData(const vector<DataVector> &newDataset)
{
    // Append and rebuild (same as KDTree).
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
    // Remove and rebuild (simple approach).
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
        cout<<endl;
    }
}

// Same search idea as KDTree; only the split rule differs.
void RPTreeIndex::searchTree(Node* node, const DataVector& testVector, int k, vector<int>& nearestIndices, vector<double>& distances) {
    if (node == nullptr) {
        return;
    }
    
    stack<Node*> path;
    Node* current = node;

    // Go down by comparing the split value.
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

    // Avoid duplicates.
    unordered_set<int> addedIndices;

    // Backtrack and check if the other side can beat the best distance.
    while (!path.empty()) {
        current = path.top();
        path.pop();

        for (int index : current->vectorIndices) {
            if (index < dataset.size()) {
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

        // Distance to the split hyperplane for this node.
        if (!current->vectorIndices.empty()) {
            double splitDistance = abs(testVector.getComponent(current->splitDim) - dataset[current->vectorIndices[0]].getComponent(current->splitDim));

            // If the other side might be closer, walk down that side as well.
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

    // Sort by distance.
    vector<pair<int, double>> nearestPairs;
    for (int i = 0; i < nearestIndices.size(); ++i) {
        nearestPairs.push_back(make_pair(nearestIndices[i], distances[i]));
    }
    sort(nearestPairs.begin(), nearestPairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
        return a.second < b.second;
    });

    // Write back ordered results.
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

    // Stop if this node is small enough.
    if (indices.size() < M)
    {
        node->isLeaf = true;
        return;
    }

    int splitDim;
    auto Rule = ChooseRule(indices, splitDim);
    
    node->vectorIndices = indices;
    // cout<<indices.size()<<endl;

    // Partition by split rule into left/right children.
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
        // Give up splitting if it would be empty on one side.
        return;
    }

    // Build children.
    node->splitDim = splitDim;
    node->leftChild = new Node();
    node->rightChild = new Node();
    buildTree(node->leftChild, leftIndices);
    buildTree(node->rightChild, rightIndices);
}

// Random split: pick a random unit direction and a small random shift, then split at the median.
function<bool(const DataVector &)> RPTreeIndex::ChooseRule(const vector<int> &indices, int &splitDim) {
     if (indices.empty()) {
        throw invalid_argument("Empty subset");
    }
    
    int numDims = dataset[indices[0]].getDimension();

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-1.0, 1.0);

    // Random unit direction in R^d.
    DataVector v(numDims);
    for (int i = 0; i < numDims; ++i) {
        v.setComponent(i, dis(gen));
    }

    // Normalize to unit length.
    v.normalize();

    // Pick a far point to scale the random shift.
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

    // Small random shift improves balance on clustered data.
    double delta = dis(gen) * 6 * sqrt(x.dist(y)) / sqrt(numDims);

    // Median along the projection.
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
    
    // Go left if dot <= median + delta.
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
        iota(indices.begin(), indices.end(), 0); // 0..n-1
        root = new Node();
        buildTree(root, indices);
    }
}

// Singleton access (one global per leaf size parameter).
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
