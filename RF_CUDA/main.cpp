#include <iostream>
#include <time.h>
#include "RandomForest.h"

using namespace std;

int main()
{
    clock_t beginTime;
        
	// Load RF from file
    printf("Loading the model...\n");
    beginTime = clock();
    
    RandomForest randomForest("./data/model.rf");
    cout << "Load time: " << float( clock () - beginTime ) /  CLOCKS_PER_SEC << "s" << endl;
    
    // Reading the datas
    printf("Loading the data...\n");
    Data data;
    data.read("./data/X_test.data");

	// Predict
    printf("Predict...\n");
    
    beginTime = clock();
    uint8_t *classes = randomForest.predict(data);
    cout << "Predict time: " << float( clock () - beginTime ) /  CLOCKS_PER_SEC << "s" << endl;
    
	// Save result in file
    ofstream pred_file("./data/cpp.pred");
    for (size_t i = 0; i < data.nSamples; ++i) {
        pred_file << int(classes[i]);
        pred_file << endl;
    }

    // Free memory
    delete [] classes;

    return 0;
}
