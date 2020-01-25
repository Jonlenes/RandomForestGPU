#ifndef UTIL_H
#define UTIL_H

#include <algorithm>
#include <iterator>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

namespace
{
    /*
     * Split a string by caracter
     */
    vector<string> splitBySpace(string &sentence) {
        istringstream iss(sentence);
        return vector<string>{istream_iterator<string>(iss),
                              istream_iterator<string>{}};
    }


	/*
	* Load a file.
	*/
    ifstream *load_file(const string &filename) {
        ifstream *file = new ifstream(filename);
        if (!*file)
        {
            cout << "File '" << filename << "' not found.\n";
            exit(-1);
        }
        return file;
    }


	/*
	* Close and free memory
	*/
    void close_file(ifstream *file) {
        file->close();
        delete file;
    }


	/*
	* Return index of max value
	*/
    int argMax(vector<double> v){
        return distance(v.begin(), max_element(v.begin(), v.end()));
    }


	/*
	* Read a list of double in file
	*/
    vector<double> getVector(ifstream &file, int size) {

        vector<double> vet;
        double value;
        for (int i = 0; i < size; ++i) {
            file >> value;
            vet.push_back(value);
        }
        return vet;
    }


}

#endif // UTIL_H
