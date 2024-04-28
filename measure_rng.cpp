#include "tsc.h"

#include <iostream>
#include <omp.h>
#include <random>
#include <string>


int main() {
  const int n = 100000;
  const int L = 20;
  auto uni = std::uniform_real_distribution<>(0., 1.);
  auto indices = std::uniform_int_distribution<>(0, L * L * L);
  auto generator = std::mt19937();
  generator.seed(__rdtsc());

  //std::random_device generator;

  int * index_sample = (int *)malloc(sizeof(int) * n); 
  double * uni_sample = (double *)malloc(sizeof(double) * n);

  INT64 start, end;

  for(int i=0;i<3 * n;++i)
      uni(generator);
  for(int i=0;i<3 * n;++i)
      indices(generator);

  uint64_t cycles = 0;
  const int NUM_TRIES = 10;
  for(int r=0;r<NUM_TRIES;++r) {
    start = start_tsc(); 
    for(int i=0;i<n;++i)
      index_sample[i] = indices(generator);    
    end = stop_tsc(start);
    cycles += end;
    for(int i=0;i<n;++i)
      index_sample[i] = 0;
    std::cout << cycles << " ";
  }
  std::cout << "\n";

  uint64_t cycles1 = cycles;

  cycles = 0;
  for(int r=0;r<NUM_TRIES;++r) {
    start = start_tsc(); 
    for(int i=0;i<n;++i)
      uni_sample[i] = uni(generator);    
    end = stop_tsc(start);
    cycles += end;
    for(int i=0;i<n;++i)
      uni_sample[i] = 0.;
    std::cout << cycles << " ";
  }
  std::cout << "\n";
  uint64_t cycles2 = cycles;

  std::cout << "index sampler: " << cycles1 / NUM_TRIES / n << "\n";
  std::cout << "uni sampler: "   << cycles2 / NUM_TRIES / n << "\n";
  
  //for(int i=0;i<n;++i)
  //  std::cout << uni(generator) << "\n";


  free(index_sample);
  free(uni_sample);


#pragma omp parallel
  {
    int * index_sample = (int *)malloc(sizeof(int) * n); 
    double * uni_sample = (double *)malloc(sizeof(double) * n);

    auto generator = std::mt19937();
    generator.seed(omp_get_wtime());

    //std::random_device generator;

    INT64 start, end;

    uint64_t cycles = 0;
    for(int r=0;r<NUM_TRIES;++r) {
      start = start_tsc(); 
      for(int i=0;i<n;++i)
        uni_sample[i] = uni(generator);    
      end = stop_tsc(start);
      cycles += end;
      for(int i=0;i<n;++i)
        uni_sample[i] = 0.;
    }

    std::string s = std::to_string(omp_get_thread_num()) + ": " + std::to_string(cycles / NUM_TRIES / n) + "\n"; 
#pragma omp critical
    std::cout << s;


    free(index_sample);
    free(uni_sample);
  }
}
