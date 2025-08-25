#include "DataVector.h"
using namespace std;
// A small vector type with common math helpers.
// Uses normal C++ value semantics and a few operator overloads.

// Create a vector with the given number of components.
DataVector::DataVector(int dimension)
{
    v.resize(dimension);
}

// Clean up. (vector already frees its memory by itself.)
DataVector::~DataVector()
{
    v.clear();
}

// Copy from another DataVector.
DataVector::DataVector(const DataVector &other) : v(other.v) {}

// Assign from another DataVector.
DataVector &DataVector::operator=(const DataVector &other)
{
    if (this != &other)
    {
        v = other.v;
    }
    return *this;
}

// Change number of components.
void DataVector::setDimension(int dimension)
{
    v.clear();
    v.resize(dimension);
}

// Component-wise sum; sizes must match.
DataVector DataVector::operator+(const DataVector &other)
{
    if (v.size() != other.v.size())
    {
        cout << "Error: vectors must have the same dimension" << endl;
        return DataVector();
    }
    else
    {
        DataVector result;
        for (int i = 0; i < v.size(); i++)
        {
            result.v.push_back(v[i] + other.v[i]);
        }
        return result;
    }
}

// Component-wise difference; sizes must match.
DataVector DataVector::operator-(const DataVector &other)
{
    if (v.size() != other.v.size())
    {
        cout << "Error: vectors must have the same dimension" << endl;
        return DataVector();
    }
    else
    {
        DataVector result;
        for (int i = 0; i < v.size(); i++)
        {
            result.v.push_back(v[i] - other.v[i]);
        }
        return result;
    }
}

// Dot product; sizes must match.
double DataVector::operator*(const DataVector &other)
{
    if (v.size() != other.v.size())
    {                                                                  // check if the vectors have the same dimension
        cout << "Error: vectors must have the same dimension" << endl;
        return 0;
    }
    else
    {
        double result = 0;
        for (int i = 0; i < v.size(); i++)
        {
            result += v[i] * other.v[i];
        }
        return result;
    }
}

// Print as <a, b, c>.
void DataVector::print() const
{
    cout << "<";
    for (int i = 0; i < v.size() - 1; i++)
    {
        cout << v[i] << ","
             << " ";
    }
    cout << v[v.size() - 1];
    cout << ">";
    cout << endl;
}

// Euclidean length of this vector. (Parameter is ignored.)
double DataVector::norm(const DataVector &other)
{
    return sqrt((*this) * (*this));
}

// Euclidean distance to another vector.
double DataVector::dist(const DataVector &other) const
{
    if (v.size() != other.v.size())
    { 
        cout<<v.size()<<" "<<other.v.size();
        throw invalid_argument("Vectors must be of the same dimension");
    }
    double distance = 0.0;
    for (size_t i = 0; i < v.size(); ++i)
    {
        distance += pow(v[i] - other.v[i], 2);
    }

    return sqrt(distance);
}

// Set one component. Warn if index is out of range.
void DataVector::setComponent(int index, double value)
{
    if (index >= 0 && index < v.size())
    {
        v[index] = value;
    }
    else
    {
        cout << "Error: Index out of range" << endl;
    }
}

// Add a new component at the end.
void DataVector::addComponent(double value)
{
    v.push_back(value);
}

// Read one component. Returns 0 if index is invalid.
double DataVector::getComponent(int index) const
{
    if (index >= 0 && index < v.size())
    {
        return v[index];
    }

    else
    {   
        return 0;
    }
}

// Number of components.
int DataVector::getDimension() const
{
    return v.size();
}

double DataVector::getMedian(int dimension) const
{
    if (dimension < 0 || dimension >= v.size())
    {
        throw out_of_range("Dimension out of range");
    }

    // Copy values and compute median without changing the original
    vector<double> dimValues(v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        dimValues[i] = v[i];
    }

    sort(dimValues.begin(), dimValues.end());

    size_t midIndex = dimValues.size() / 2;
    if (dimValues.size() % 2 == 1)
    {
        return dimValues[midIndex];
    }
    else
    {
        return (dimValues[midIndex - 1] + dimValues[midIndex]) / 2.0;
    }
}

void DataVector::readDataset(const string &filename, vector<DataVector> &dataset)
{
    // Read a CSV where each line is one vector of numbers.
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Unable to open file " << filename << endl;
        return;
    }

    string line;
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

bool DataVector::operator==(const DataVector &other) const
{
    // Element-wise equality (exact float compare on purpose).
    for (int i = 0; i < v.size(); ++i)
    {
        if (v[i] != other.v[i])
        {
            return false;
        }
    }

    return true;
}

// Fill components with random values in [-1, 1].
void DataVector::randomize()// randomly assign values between -1 and 1
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-1.0, 1.0);

    for (double &component : v)
    {
        component = dis(gen);
    }
}

double DataVector::dot(const DataVector &other) const
{
    double result = 0.0;
    int dimension = min(getDimension(), other.getDimension());
    for (int i = 0; i < dimension; ++i)
    {
        result += v[i] * other.getComponent(i);
    }
    return result;
}

void DataVector::normalize() 
{
    // Scale vector to unit length (skips if norm is 0).
    double norm = 0.0;
    for (double component : v)
    {
        norm += component * component;
    }
    norm = sqrt(norm);
    for (double &component : v)
    {
        component /= norm;
    }
}
