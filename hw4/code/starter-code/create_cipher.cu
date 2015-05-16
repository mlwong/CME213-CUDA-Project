#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
// You may include other thrust headers if necessary.

#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "test_macros.h"

// You will need to call these functors from
// thrust functions in the code do not create new ones

// returns true if the char is not a lowercase letter
struct isnot_lowercase_alpha : thrust::unary_function<unsigned char, bool> {
  // TODO
  
  __host__ __device__
  bool operator()(const unsigned char &character)
  {
    if (character < 'a' || character > 'z')
    {
      return true;
    }
    return false;
  }
  
};

// convert an uppercase letter into a lowercase one
// do not use the builtin C function or anything from boost, etc.
struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char> {
  // TODO
  
  __host__ __device__
  unsigned char operator()(const unsigned char &character)
  {
    if (character >= 'A' && character <= 'Z')
    {
      return (character + 32);
    }
    else
    {
      return character;
    }
  }
  
};

// apply a shift with appropriate wrapping
struct apply_shift : thrust::binary_function<unsigned char, int,
                                             unsigned char> {
  // TODO
  
  unsigned int *shift_amounts;
  unsigned int period;
  
  __host__ __device__
  apply_shift(unsigned int *shift_amounts_input, unsigned int period_input):
                shift_amounts(shift_amounts_input),
                period(period_input)
  {}
  
  __host__ __device__
  unsigned char operator()(const unsigned char &character, const unsigned int &position)
  {
    unsigned char converted_character = character + shift_amounts[position%period];
    
    if (converted_character > 122)
    {
      converted_character -= 26;
    }
    
    return converted_character;    
  }
  
};

// Returns a vector with the top 5 letter frequencies in text.
std::vector<double> getLetterFrequencyCpu(
    const std::vector<unsigned char> &text) {
  std::vector<unsigned int> freq(256);
  for (unsigned int i = 0; i < text.size(); ++i)
    freq[tolower(text[i])]++;

  unsigned int sum_chars = 0;
  for (unsigned char c = 'a'; c <= 'z'; ++c)
    sum_chars += freq[c];

  std::vector<double> freq_alpha_lower;
  for (unsigned char c = 'a'; c <= 'z'; ++c) {
    if (freq[c] > 0)
      freq_alpha_lower.push_back(freq[c] / static_cast<double>(sum_chars));
  }

  std::sort(freq_alpha_lower.begin(), freq_alpha_lower.end(),
            std::greater<double>());
  freq_alpha_lower.resize(min(static_cast<int>(freq_alpha_lower.size()), 5));

  return freq_alpha_lower;
}

// Print the top 5 letter frequencies and them.
std::vector<double> getLetterFrequencyGpu(
    const thrust::device_vector<unsigned char> &text) {
  std::vector<double> freq_alpha_lower;
  // WARNING: make sure you handle the case of not all letters appearing
  // in the text.

  // TODO calculate letter frequency
  
  thrust::device_vector<unsigned char> characters = text;
  
  // assume the text has all 26 letters
  // find the number of occurence of each letter
  
  thrust::device_vector<unsigned char> histogram_keys(26);
  thrust::device_vector<int> histogram_values(26);
  
  thrust::sort(characters.begin(), characters.end());
  
  typedef thrust::device_vector<unsigned char>::iterator devIt_1;
  typedef thrust::device_vector<int>::iterator devIt_2;
  
  thrust::pair<devIt_1, devIt_2> endIterators =
    thrust::reduce_by_key(characters.begin(),
                          characters.end(),
                          thrust::make_constant_iterator(1),
                          histogram_keys.begin(),
                          histogram_values.begin());
  
  int num_letters = endIterators.first - histogram_keys.begin();
  
  // sort the letters according to descending order of number of
  // occurence of letters
  
  thrust::sort_by_key(histogram_values.begin(),
                      histogram_values.end(),
                      histogram_keys.begin(),
                      thrust::greater<int>());
  
  // output the top 5 letters along with their frequencies
  
  std::cout << "Top 5 Letter Frequencies" << std::endl;
  std::cout << "-------------" << std::endl;
    
  for (unsigned int i = 0; i < min(num_letters, 5); i++)
  {
    freq_alpha_lower.push_back(((double) histogram_values[i])/((double) text.size()));
    std::cout << histogram_keys[i]
              << " "
              << freq_alpha_lower[i]
              << std::endl;
  }
  
  return freq_alpha_lower;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Didn't supply plain text and period!" << std::endl;
    return 1;
  }

  std::ifstream ifs(argv[1], std::ios::binary);
  if (!ifs.good()) {
    std::cerr << "Couldn't open text file!" << std::endl;
    return 1;
  }

  unsigned int period = atoi(argv[2]);

  if (period < 4) {
    std::cerr << "Period must be at least 4!" << std::endl;
    return 1;
  }

  // load the file into text
  std::vector<unsigned char> text;

  ifs.seekg(0, std::ios::end); // seek to end of file
  int length = ifs.tellg();    // get distance from beginning
  ifs.seekg(0, std::ios::beg); // move back to beginning

  text.resize(length);
  ifs.read((char *)&text[0], length);

  ifs.close();

  thrust::device_vector<unsigned char> text_clean;
  
  // TODO: sanitize input to contain only a-z lowercase (use the
  // isnot_lowercase_alpha functor), calculate the number of characters
  // in the cleaned text and put the result in text_clean, make sure to
  // resize text_clean to the correct size!
  int numElements = -1;
  
  thrust::device_vector<unsigned char> d_text = text;
  text_clean.resize(d_text.size());
  
  thrust::device_vector<unsigned char>::iterator endIterator =
    thrust::remove_copy_if(thrust::make_transform_iterator(d_text.begin(), upper_to_lower()),
                           thrust::make_transform_iterator(d_text.end(), upper_to_lower()),
                           text_clean.begin(),
                           isnot_lowercase_alpha());
  
  numElements = endIterator - text_clean.begin();
  
  text_clean.resize(numElements);
  
  std::cout << "\nBefore ciphering!" << std::endl << std::endl;
  std::vector<double> letterFreqGpu = getLetterFrequencyGpu(text_clean);
  std::vector<double> letterFreqCpu = getLetterFrequencyCpu(text);
  bool success = true;
  EXPECT_VECTOR_EQ_EPS(letterFreqCpu, letterFreqGpu, 1e-14, &success);
  PRINT_SUCCESS(success);

  thrust::device_vector<unsigned int> shifts(period);
  
  // TODO fill in shifts using thrust random number generation (make sure
  // not to allow 0-shifts, this would make for rather poor encryption).
  
  thrust::default_random_engine rng(123); // use seed of 123
  thrust::uniform_int_distribution<unsigned int> u_dist(1, 25);
  
  for (unsigned int i = 0; i < period; i++)
  {
    shifts[i] = u_dist(rng);
  }
  
  std::cout << "\nEncryption key: ";
  for (int i = 0; i < period; ++i)
    std::cout << static_cast<char>('a' + shifts[i]);
  std::cout << std::endl;

  thrust::device_vector<unsigned char> device_cipher_text(numElements);

  // TODO: Apply the shifts to text_clean and place the result in
  // device_cipher_text.
  
  apply_shift as(thrust::raw_pointer_cast(&shifts[0]), period);
  
  thrust::transform(text_clean.begin(),
                    text_clean.end(),
                    thrust::make_counting_iterator(0),
                    device_cipher_text.begin(),
                    as);
  
  thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;
  
  std::cout << "After ciphering!" << std::endl << std::endl;
  getLetterFrequencyGpu(device_cipher_text);

  std::ofstream ofs("cipher_text.txt", std::ios::binary);

  ofs.write((char *)&host_cipher_text[0], numElements);

  ofs.close();

  return 0;
}
