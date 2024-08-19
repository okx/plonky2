CXX := g++

CXXFLAGS := -Wall -Wno-error=sign-compare -std=c++17 -O3 -march=native -fopenmp -fPIC

all: libmerkle_avx512.a

lib: libmerkle_avx512.a

libmerkle_avx512.so: merkle.h merkle.cpp
	$(CXX) $(CXXFLAGS) -DTESTING -fPIC -D__USE_AVX__ -D__AVX512__ -D__NO_GMP__ -mavx2 -mavx512f -shared -o $@ merkle.cpp

libmerkle_avx512.a: merkle.h merkle.cpp
	$(CXX) $(CXXFLAGS) -DTESTING -fPIC -D__USE_AVX__ -D__AVX512__ -D__NO_GMP__ -mavx2 -mavx512f -c merkle.cpp
	ar rvs $@ merkle.o

clean:
	rm -f libmerkle_avx512.a libmerkle_avx512.so merkle.o