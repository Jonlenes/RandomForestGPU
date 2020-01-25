objects = main.o DProcess.o Data.o RandomForest.o

all: $(objects)
	nvcc -std=c++11 $(objects) -o app

%.o: %.cpp
	nvcc -g -std=c++11 -x cu -I. -dc $< -o $@

clean:
	rm -f *.o app