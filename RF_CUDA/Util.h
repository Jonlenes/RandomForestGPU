#ifndef UTIL_H
#define UTIL_H

#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>

using namespace std;

namespace
{

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
    * Close file and free memory
	*/
    void close_file(ifstream *file) {
        file->close();
        delete file;
    }


	/*
	* Read a list of double in file
	*/
    float *getArray(ifstream &file, size_t size) {
        float *vet = new float[size];
        float value;
        for (size_t i = 0; i < size; ++i) {
            file >> value;
            vet[i] = value;
        }
        return vet;
    }

}

#endif // UTIL_H
