#include "VectorDataset.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono> // include the chrono library to measure the time taken to find the nearest neighbors
using namespace std;
// Standalone KNN driver for a dense, small dataset.
// kNearestNeighbors builds a distance list to every row, sorts, and returns the top-k rows.

VectorDataset kNearestNeighbors(VectorDataset &dataset, DataVector &testVector, int k)
{
    // Guard empty dataset
    if (dataset.size() == 0)
    {
        cout << "Error: empty dataset" << endl;
        return VectorDataset();
    }

    else
    {
        // Pair of (distance, rowIndex)
        vector<pair<double, int>> distances;

        for (int i = 0; i < dataset.size(); i++)
        {
            // Accumulate distances to every dataset vector
            distances.push_back(make_pair(testVector.dist(dataset.getVector(i)), i));
        }

        // Sort ascending by distance
        sort(distances.begin(), distances.end());

        // Collect top-k rows
        VectorDataset topkNeighbours;

        for (int i = 0; i < k; i++)
        {
            topkNeighbours.push_back(dataset.getVector(distances[i].second));
        }

        return topkNeighbours;
    }
}

// Minimal CLI to demonstrate KNN over a file-backed dataset and a small test set.

int main()
{
    VectorDataset dataset;
    cout << "Your Dataset is being read..." << endl;
    dataset.readDataset("testing.csv");
    // dataset.readDataset("fmnist-test.csv");
    cout << "Dataset read successfully!" << endl;

    // Ask for the dimension of the test vector from the user.
    // The dimension should be equal to the number of columns in the dataset.
    // int dimension=7;
    // cout << "Enter the dimension of the test vector.";
    // cout << "The dimension should be equal to the number of columns in the dataset:" << endl;
    // cin >> dimension;

    // Create a DataVector object to store the test vector.
    // The dimension of the test vector is set to the value entered by the user.
    // The components of the test vector are set by the user.
    // The components of the test vector are set to 5.8 for testing purposes.
    // DataVector testVector(dimension);
    // cout << "Enter the values of the test vector:" << endl;

    // Read test vectors from file
    VectorDataset testvectorDataset;
    testvectorDataset.readDataset("test-vector.csv");

    // Ask for the components of the test vector from the user.
    // The components are set using the setComponent method of the DataVector class.
    // for (int i = 0; i < dimension; i++)
    // {
    //     // cout << "Enter the value of the " << i + 1 << "th dimension: ";
    //     // double x;
    //     // cin >> x;
    //     // testVector.setComponent(i, x);
    //     testVector.setComponent(i, 5.8);
    // }

    // Number of nearest neighbours to return (fixed here, could be interactive)
    int k=4;
    cout << "Enter the value of k i.e. the number of nearest neighbours you want to find: ";
    // cin >> k;

    // Time the end-to-end KNN run for simple performance feedback
    int number=1;
    cout<<"Enter the number of test vectors you want to test:";
    // cin>>number;
    cout << "Finding the nearest neighbours..." << endl;

    auto start = chrono::high_resolution_clock::now();
    VectorDataset nearestNeighbors;


    if(number>testvectorDataset.size()){
        cout << "Error: Number of test vectors is greater than the size of the dataset" << endl;
        return 0;
    }
   

    for (int i = 0; i < number; i++)
    {   
        DataVector testVector = testvectorDataset.getVector(i);
        nearestNeighbors = kNearestNeighbors(dataset, testVector, k);
    }
    // DataVector testVector(dimension);

    auto end = chrono::high_resolution_clock::now();
    cout << "Nearest neighbours found successfully!" << endl;

    cout << "Nearest neighbour search took: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    // Print the nearest neighbors to the console
    cout << "Nearest Neighbors are:" << endl;
    for (int i = 0; i < nearestNeighbors.size(); ++i)
    {
        cout << "Nearest Neighbour " << i + 1 << ": ";
        nearestNeighbors.getVector(i).print();
        cout << endl;
    }

    return 0;
}
