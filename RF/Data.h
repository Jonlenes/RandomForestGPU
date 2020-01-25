#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <functional>
#include "Util.h"


using namespace std;

class Data {
private:
    vector<vector<double>> features;
    vector<int> samplesVec;

	int featureSize = 0;
    int samplesSize = 0;
    
public:
    Data();

    void read(const string &filename);

    double getFeature(int sampleIndex, int featureIndex);

    int getSampleSize();

    int getFeatureSize();

    vector<int> generateSample(int &num);

};

#endif //DATA_H
