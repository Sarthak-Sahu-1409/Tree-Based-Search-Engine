#include <bits/stdc++.h>
#include <random>
#include "TreeIndex.h"
using namespace std;

// Initialize random number generator.
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(-1.0, 1.0);

// --- KD Tree implementation ---

void KDTreeIndex::AddData(const vector<DataVector> &newDataset)
{
    // dataset.clear();
    dataset.insert(dataset.end(), newDataset.begin(), newDataset.end());
    // newDataset[1].print();
    
    if (root)
    {
        delete root;
        root = nullptr;
    }
    MakeTree();
}

void KDTreeIndex::RemoveData(const vector<DataVector> &dataToRemove)
{
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

void KDTreeIndex::searchTree(Node* node, const DataVector& testVector, int k, vector<int>& nearestIndices, vector<double>& distances) {
    if (node == nullptr) {
        return;
    }
    
    stack<Node*> path;
    Node* current = node;

    // Traverse down the tree to find the leaf node closest to the test vector
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

    // Keep track of the indices that have been added to the nearest neighbors
    unordered_set<int> addedIndices;

    // Traverse back up the tree to find potential nearest neighbors
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

        // Calculate the distance between the test vector and the splitting hyperplane
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

    // Sort the nearest neighbors by distance
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

void KDTreeIndex::buildTree(Node *&node, const std::vector<int> &indices)
{
    if (indices.empty())
    {
        return;
    }

    // If the number of vectors in this node is less than M, make it a leaf
    if (indices.size() < M)
    {
        node->isLeaf = true;
        return;
    }

    int splitDim;
    auto Rule = ChooseRule(indices, splitDim);
    
    node->vectorIndices = indices;
    
    // Partition the indices based on the split rule
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
        // Handle the case where one of the partitions is empty
        return;
    }

    // Recursively build the left and right subtrees
    node->splitDim = splitDim;
    node->leftChild = new Node();
    node->rightChild = new Node();
    buildTree(node->leftChild, leftIndices);
    buildTree(node->rightChild, rightIndices);
}

std::function<bool(const DataVector &)> KDTreeIndex::ChooseRule(const std::vector<int> &indices, int &splitDim) {
    if (indices.empty()) {
        throw std::invalid_argument("Empty subset");
    }
    
    int numDims = dataset[indices[0]].getDimension();
    vector<double> maxVals(numDims, numeric_limits<double>::lowest());
    vector<double> minVals(numDims, numeric_limits<double>::max());

    // Calculate the minimum and maximum values for each dimension
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

    // Calculate the spread for each dimension and choose the dimension with the maximum spread
    for (int i = 0; i < numDims; ++i) {
        double spread = maxVals[i] - minVals[i];
        if (spread > maxSpread) {
            maxSpread = spread;
            splitDim = i;
        }
    }

    // Calculate the median value for the chosen dimension
    vector<double> dimVals;
    for (int index : indices) {
        dimVals.push_back(dataset[index].getComponent(splitDim));
    }
    sort(dimVals.begin(), dimVals.end());
    double median = dimVals[dimVals.size() / 2];

    // Return the splitting rule as a lambda function
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
        iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, 2, ..., n-1
        root = new Node();
        buildTree(root, indices);
    }
}

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

void RPTreeIndex::AddData(const vector<DataVector> &newDataset)
{
    // dataset.clear();
    dataset.insert(dataset.end(), newDataset.begin(), newDataset.end());
    // newDataset[1].print();
    
    if (root)
    {
        delete root;
        root = nullptr;
    }
    MakeTree();
}

void RPTreeIndex::RemoveData(const vector<DataVector> &dataToRemove)
{
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

void RPTreeIndex::searchTree(Node* node, const DataVector& testVector, int k, vector<int>& nearestIndices, vector<double>& distances) {
    if (node == nullptr) {
        return;
    }
    
    stack<Node*> path;
    Node* current = node;

    // Traverse down the tree to find the leaf node closest to the test vector
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

    // Keep track of the indices that have been added to the nearest neighbors
    unordered_set<int> addedIndices;

    // Traverse back up the tree to find potential nearest neighbors
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

        // Calculate the distance between the test vector and the splitting hyperplane
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

    // Sort the nearest neighbors by distance
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

void RPTreeIndex::buildTree(Node *&node, const std::vector<int> &indices)
{
    if (indices.empty())
    {
        return;
    }

    // If the number of vectors in this node is less than M, make it a leaf
    if (indices.size() < M)
    {
        node->isLeaf = true;
        return;
    }

    int splitDim;
    auto Rule = ChooseRule(indices, splitDim);
    
    node->vectorIndices = indices;
    // cout<<indices.size()<<endl;

    // Partition the indices based on the split rule
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
        // Handle the case where one of the partitions is empty
        return;
    }

    // Recursively build the left and right subtrees
    node->splitDim = splitDim;
    node->leftChild = new Node();
    node->rightChild = new Node();
    buildTree(node->leftChild, leftIndices);
    buildTree(node->rightChild, rightIndices);
}

std::function<bool(const DataVector &)> RPTreeIndex::ChooseRule(const std::vector<int> &indices, int &splitDim) {
     if (indices.empty()) {
        throw std::invalid_argument("Empty subset");
    }
    
    int numDims = dataset[indices[0]].getDimension();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    // Choose a random unit direction v
    DataVector v(numDims);
    for (int i = 0; i < numDims; ++i) {
        v.setComponent(i, dis(gen));
    }

    // Normalize to get a unit vector
    v.normalize();
    // v.print();

    // Find the farthest point y from any point x in S
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

    // Calculate δ uniformly at random
    double delta = dis(gen) * 6 * sqrt(x.dist(y)) / sqrt(numDims);

    // Calculate the median dot product
    double medianDotProduct = 0.0;
    std::vector<double> dotProducts;
    for (int index : indices) {
        dotProducts.push_back(dataset[index].dot(v));
    }
    std::sort(dotProducts.begin(), dotProducts.end());
    int n = dotProducts.size();
    if (n % 2 == 0) {
        medianDotProduct = (dotProducts[n / 2 - 1] + dotProducts[n / 2]) / 2.0;
    } else {
        medianDotProduct = dotProducts[n / 2];
    }
    
    // Return the splitting rule as a lambda function.
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
        iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, 2, ..., n-1
        root = new Node();
        buildTree(root, indices);
    }
}

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
