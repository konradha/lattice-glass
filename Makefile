CC=clang++-17
CFLAGS =-I. -ftree-vectorize -pedantic -ffast-math -march=native -O3 -Wall -fopenmp -Wunknown-pragmas  -lm -lstdc++ -std=c++17
TARGET = to_omp
SRC = sim_omp.cpp
RUN_SCRIPT = python run_local.py


$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	$(RUN_SCRIPT)

clean:
	rm $(TARGET)
