#include "Data.h"


Data::Data() {
    this->featureSize = -1;
}


/*
* Read the data
*/
void Data::read(const string &filename) {
    ifstream &file = *load_file(filename);
    string str;

    while (getline(file, str)) {
        vector<string> results = splitBySpace(str);
        if (this->featureSize == -1)
            this->featureSize = results.size();

        vector<double> sample(this->featureSize, 0);

        for (int i = 0; i < results.size(); i++)
            sample[i] = atof(results[i].c_str());

        this->features.push_back(sample);
        samplesVec.push_back(this->samplesSize++);
    }

    close_file(&file);
}


/*
* Return a feature
*/
double Data::getFeature(int sampleIndex, int featureIndex) {
    return features[sampleIndex][featureIndex];
}


/*
* Number of sample
*/
int Data::getSampleSize() {
    return features.size();
}


/*
* Number de feaures
*/
int Data::getFeatureSize() {
    return featureSize;
}


/*
* Return samples
*/
vector<int> Data::generateSample(int &num) {
    random_shuffle(samplesVec.begin(), samplesVec.end());
    return vector<int>(samplesVec.begin(), samplesVec.begin() + num);
}
