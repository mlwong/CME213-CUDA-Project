#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <assert.h>
#include <omp.h>
#include <stdlib.h>

#include "tests.h"

typedef unsigned int uint;

const uint kSizeTestVector = 4000000;
const uint kSizeMask = 16; // must be a divider of 32 for this program to work correctly
const uint kRandMax = 1 << 31;
const uint kNumBitsUint = 32;

/* Function: computeBlockHistograms
 * --------------------------------
 * Splits keys into numBlocks and computes an histogram with numBuckets buckets 
 * Remember that numBuckets and numBits are related; same for blockSize and numBlocks.
 * Should work in parallel.
 */
std::vector<uint> computeBlockHistograms(const std::vector<uint>& keys, uint numBlocks, uint numBuckets,
                                         uint numBits, uint startBit, uint blockSize) {
  std::vector<uint> blockHistograms(numBlocks * numBuckets, 0);
  // TODO
  
  uint mask = numBuckets - 1;
  
#pragma omp parallel for
  for (uint n = 0; n < numBlocks; n++)
  {
    for (uint i = n*blockSize; (i < (n + 1)*blockSize && i < keys.size()); i++)
    {
      uint bucket = (keys[i] >> startBit) & mask;
      ++blockHistograms[n*numBuckets + bucket];
    }
  } 
  
  return blockHistograms;
}

/* Function: reduceLocalHistoToGlobal
 * ----------------------------------
 * Takes as input the local histogram of size numBuckets * numBlocks and "merges"
 * them into a global histogram of size numBuckets.
 */ 
std::vector<uint> reduceLocalHistoToGlobal(const std::vector<uint>& blockHistograms, uint numBlocks, uint numBuckets) {
  std::vector<uint> globalHisto(numBuckets, 0);
  // TODO
  
  for (uint n = 0; n < numBlocks; n++)
  {
    for (uint i = 0; i < numBuckets; i++)
    {
      globalHisto[i] += blockHistograms[n*numBuckets + i];
    }
  }
  
  return globalHisto;
}

/* Function: computeBlockExScanFromGlobalHisto 
 * -------------------------------------------
 * Takes as input the globalHistoExScan that contains the global histogram after the scan
 * and the local histogram in blockHistograms. Returns a local histogram that will be used
 * to populate the sorted array.
 */
std::vector<uint> computeBlockExScanFromGlobalHisto(uint numBuckets, uint numBlocks,
                                                    const std::vector<uint>& globalHistoExScan,
                                                    const std::vector<uint>& blockHistograms) { 
  std::vector<uint> blockExScan(numBuckets * numBlocks, 0);
  // TODO
  
  for (uint i = 0; i < numBuckets; i++)
  {
    blockExScan[i] = globalHistoExScan[i];
  }
  
  for (uint n = 1; n < numBlocks; n++)
  {
    for (uint i = 0; i < numBuckets; i++)
    {
      blockExScan[n*numBuckets + i] = blockExScan[(n - 1)*numBuckets + i] + blockHistograms[(n - 1)*numBuckets + i];
    }
  }
  
  return blockExScan;
}

/* Function: populateOutputFromBlockExScan
 * ---------------------------------------
 * Takes as input the blockExScan produced by the splitting of the global histogram
 * into blocks and populates the vector sorted.
 */
void populateOutputFromBlockExScan(const std::vector<uint>& blockExScan, uint numBlocks, uint numBuckets, uint startBit,
                                   uint numBits, uint blockSize, const std::vector<uint>& keys, std::vector<uint> &sorted) {
  // TODO
  
  uint mask = numBuckets - 1;
  
#pragma omp parallel for
  for (uint n = 0; n < numBlocks; n++)
  {
    std::vector<uint> bucketCounts(numBuckets, 0);
    for (uint i = n*blockSize; (i < (n + 1)*blockSize && i < keys.size()); i++)
    {
      uint bucket = (keys[i] >> startBit) & mask;
      sorted[blockExScan[n*numBuckets + bucket] + bucketCounts[bucket]] = keys[i];
      ++bucketCounts[bucket];
    }
  }
}

/* Function: scanGlobalHisto
 * -------------------------
 * This function should simply scan the global histogram.
 */
std::vector<uint> scanGlobalHisto(const std::vector<uint>& globalHisto, uint numBuckets) {
  std::vector<uint> globalHistoExScan(numBuckets, 0);
  // TODO
  
  for (uint i = 1; i < numBuckets; i++)
  {
    globalHistoExScan[i] = globalHistoExScan[i - 1] + globalHisto[i - 1];
  }
  
  return globalHistoExScan;
}

/* Function: radixSortParallelPass
 * -------------------------------
 * A pass of radixSort on numBits starting after startBit.
 */
void radixSortParallelPass(std::vector<uint> &keys, std::vector<uint> &sorted, uint numBits, uint startBit,
                           uint blockSize) {
  uint numBuckets = 1 << numBits;
  // Choose numBlocks so that numBlocks * blockSize is always greater than keys.size().
  uint numBlocks = (keys.size() + blockSize - 1) / blockSize;
  
  // go over each block and compute its local histogram
  std::vector<uint> blockHistograms = computeBlockHistograms(keys, numBlocks, numBuckets, numBits, startBit, blockSize);
  
  // first reduce all the local histograms into a global one
  std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms, numBlocks, numBuckets);

  // now we scan this global histogram
  std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);
  
  // now we do a local histogram in each block and add in the global value to get global position
  std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets, numBlocks, globalHistoExScan, blockHistograms); 
  
  // populate the sorted vector
  populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, startBit, numBits, blockSize, keys, sorted);
}

int radixSortParallel(std::vector<uint> &keys, std::vector<uint> &keys_tmp, uint numBits) {
  for (uint startBit = 0; startBit < kNumBitsUint; startBit += 2 * numBits) {
    radixSortParallelPass(keys, keys_tmp, numBits, startBit, keys.size() / 8); 
    radixSortParallelPass(keys_tmp, keys, numBits, startBit + numBits, keys.size() / 8); 
  }
  return 0;
}

void radixSortSerialPass(std::vector<uint> &keys, std::vector<uint> &keys_radix, uint startBit, uint numBits) {
  uint numBuckets = 1 << numBits;
  uint mask = numBuckets - 1;

  //compute the frequency histogram
  std::vector<uint> histogramRadixFrequency(numBuckets);
  for (uint i = 0; i < keys.size(); ++i) {
    uint key = (keys[i] >> startBit) & mask;
    ++histogramRadixFrequency[key];
  }

  //now scan it
  std::vector<uint> exScanHisto(numBuckets, 0);
  for (uint i = 1; i < numBuckets; ++i) {
    exScanHisto[i] = exScanHisto[i - 1] + histogramRadixFrequency[i-1];
    histogramRadixFrequency[i - 1] = 0;
  }

  histogramRadixFrequency[numBuckets - 1] = 0;

  //now add the local to the global and scatter the result
  for (uint i = 0; i < keys.size(); ++i) {
    uint key = (keys[i] >> startBit) & mask;

    uint localOffset = histogramRadixFrequency[key]++;
    uint globalOffset = exScanHisto[key] + localOffset;

    keys_radix[globalOffset] = keys[i];
  }
}

int radixSortSerial(std::vector<uint> &keys, std::vector<uint> &keys_radix, uint numBits) {
  assert(numBits <= 16);
  for (uint startBit = 0; startBit < 32; startBit += 2 * numBits) {
    radixSortSerialPass(keys, keys_radix, startBit, numBits);
    radixSortSerialPass(keys_radix, keys, startBit+numBits, numBits);
  }
  return 0;
}

void initializeRandomly(std::vector<uint>& keys) { 
  std::default_random_engine generator;
  std::uniform_int_distribution<uint> distribution(0, kRandMax);
  for (uint i = 0; i < keys.size(); ++i)
    keys[i] = distribution(generator);
}

int main() {
  Test1();
  Test2();
  Test3();
  Test4();
  Test5();
 
  // Initialize Variables
  std::vector<uint> keys_stl(kSizeTestVector);
  initializeRandomly(keys_stl);
  std::vector<uint> keys_serial = keys_stl;
  std::vector<uint> keys_parallel = keys_stl;
  std::vector<uint> temp_keys(kSizeTestVector);

#ifdef QUESTION6
  std::vector<uint> keys_parallels[7];
  for (int i = 0; i < 7; i++)
  {
    keys_parallels[i] = keys_stl;
  }
#endif
  
  // stl sort
  double startstl = omp_get_wtime();
  std::sort(keys_stl.begin(), keys_stl.end());
  double endstl = omp_get_wtime();
 
  // serial radix sort
  double startRadixSerial = omp_get_wtime();
  radixSortSerial(keys_serial, temp_keys, kSizeMask);
  double endRadixSerial = omp_get_wtime();

  bool success = true;
  EXPECT_VECTOR_EQ(keys_stl, keys_serial, &success);
  if (success)
    std::cout << "Serial Radix Sort is correct" << std::endl;

  // parallel radix sort
  double startRadixParallel = omp_get_wtime();
  radixSortParallel(keys_parallel, temp_keys, kSizeMask);
  double endRadixParallel = omp_get_wtime();

  success = true;
  EXPECT_VECTOR_EQ(keys_stl, keys_parallel, &success);
  if (success)
    std::cout << "Parallel Radix Sort is correct" << std::endl;

  std::cout << "stl: " << endstl - startstl << std::endl;
  std::cout << "serial radix: " << endRadixSerial - startRadixSerial << std::endl;
  std::cout << "parallel radix: " << endRadixParallel - startRadixParallel << std::endl;

#ifdef QUESTION6
  std::cout << "Run parallel radix for 1,2,4,8,16,32,64 threads" << std::endl;
  for (int i = 0; i < 7; i++)
  {
    omp_set_num_threads (1 << i);
    double startRadixParallel = omp_get_wtime ();
    radixSortParallel(keys_parallels[i], temp_keys, kSizeMask);
    double endRadixParallel = omp_get_wtime ();
    std::cout << (1 << i) << " threads: " << endRadixParallel - startRadixParallel << std::endl;
  }
#endif

  return 0;
}

