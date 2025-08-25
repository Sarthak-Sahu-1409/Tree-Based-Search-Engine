#include <iostream>
#include "VectorDataset.h"
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;
// VectorDataset: thin container around vector<DataVector> with simple CSV ingestion.
// OOP highlights: value semantics (copy/assign), cohesive responsibility (own/return DataVectors),
// and a small API used by KNN and tree builders.

// Constructor: ensure container starts empty.
VectorDataset::VectorDataset()
{
    dataset.clear();
}

// Destructor: no manual resource management needed.
VectorDataset::~VectorDataset()
{
    dataset.clear();
}

// Copy constructor (OOP: value semantics)
VectorDataset::VectorDataset(const VectorDataset &other) : dataset(other.dataset) {}

// Assignment operator (OOP: value semantics)
VectorDataset &VectorDataset::operator=(const VectorDataset &other)
{
    if (this != &other)
    {
        dataset = other.dataset;
    }
    return *this;
}

// Read a file where each line is a CSV of numeric values forming one DataVector.
void VectorDataset::readDataset(const string filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Unable to open file " << filename << endl;
        return;
    }

    string line;
    int i = 0;
    while (getline(file, line))
    {
        istringstream iss(line);

        DataVector dataVector;

        double value;
        while (iss >> value)
        {
            dataVector.addComponent(value);
            char comma;
            if (iss >> comma && comma != ',')
            {
                cerr << "Error: Invalid CSV format" << endl;
                dataset.clear();
                break;
            }
        }

        dataset.push_back(dataVector);
    }

    file.close();
}

// Accessor: returns a copy by value (callers typically consume it).
DataVector VectorDataset::getVector(int index)
{
    return dataset[index];
}

// Number of vectors loaded.
int VectorDataset::size()
{
    return dataset.size();
}

// Remove all vectors.
void VectorDataset::clear()
{
    dataset.clear();
}

// Append a vector to the dataset.
void VectorDataset::push_back(const DataVector &dataVector)
{
    dataset.push_back(dataVector);
}
