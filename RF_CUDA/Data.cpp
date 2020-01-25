#include "Data.h"


Data::Data() {
    nFeatures = 0;
    nSamples = 0;
    features = nullptr;
}


Data::~Data() {
    if (features) 
        delete [] features;
}


/*
* Read the data
*/
void Data::read(const string &filename) {
    ifstream &file = *load_file(filename);
    string str;
    float number;

    file >> nSamples;
    file >> nFeatures;

    features = new float[nSamples * nFeatures];

    for (int i = 0; i < nSamples; ++i) {
        for (int j = 0; j < nFeatures; j++) {
            file >> number;
            features[i * nFeatures + j] = number;
        }
    }

    close_file(&file);
}