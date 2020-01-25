#ifndef DATA_H
#define DATA_H

#include "Util.h"


using namespace std;

class Data {
public:
    float *features;

	int nFeatures;
    int nSamples;
    
public:
    Data();
    ~Data();

    void read(const string &filename);
};

#endif //DATA_H
