CXX := g++

CXXFLAGS := -Wall -Wno-error=sign-compare -std=c++17 -O3 -march=native -fopenmp -fPIC

all: libmerkle_avx512.so

lib: libmerkle_avx512.so

libmerkle_avx512.so: merkle.h merkle.cpp
	$(CXX) $(CXXFLAGS) -DTESTING -fPIC -D__USE_AVX__ -D__AVX512__ -D__NO_GMP__ -mavx2 -mavx512f -shared -o $@ merkle.cpp
