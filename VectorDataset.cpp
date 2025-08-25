#include <iostream>
#include "VectorDataset.h"
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;
// Holds a list of DataVector rows and knows how to read them from CSV.

// Start empty.
VectorDataset::VectorDataset()
{
    dataset.clear();
}

// Nothing special to clean up.
VectorDataset::~VectorDataset()
{
    dataset.clear();
}

// Copy whole dataset.
VectorDataset::VectorDataset(const VectorDataset &other) : dataset(other.dataset) {}

// Assign whole dataset.
VectorDataset &VectorDataset::operator=(const VectorDataset &other)
{
    if (this != &other)
    {
        dataset = other.dataset;
    }
    return *this;
}

// Read a file where each line is a CSV of numbers forming one DataVector.
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

// Get a row by index (by value).
DataVector VectorDataset::getVector(int index)
{
    return dataset[index];
}

// Number of rows.
int VectorDataset::size()
{
    return dataset.size();
}

// Remove everything.
void VectorDataset::clear()
{
    dataset.clear();
}

// Append one row.
void VectorDataset::push_back(const DataVector &dataVector)
{
    dataset.push_back(dataVector);
}
