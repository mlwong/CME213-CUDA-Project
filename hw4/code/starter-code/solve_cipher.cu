#include <vector>
#include <fstream>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>

#include "strided_range_iterator.h"

// You will need to call these functors from thrust functions in the code
// do not create new ones

// this can be the same as in create_cipher.cu
struct apply_shift : thrust::binary_function<unsigned char, int,
                                             unsigned char> {
  // TODO
  
  int *shift_amounts;
  int period;
  
  __host__ __device__
  apply_shift(int *shift_amounts_input, int period_input):
                shift_amounts(shift_amounts_input),
                period(period_input)
  {}
  
  __host__ __device__
  unsigned char operator()(const unsigned char &character, const unsigned int &position)
  {
    unsigned char converted_character = character + shift_amounts[position%period];
    
    if (converted_character > 'z')
    {
      converted_character -= 26;
    }
    else if (converted_character < 'a')
    {
      converted_character += 26;
    }
    
    return converted_character;    
  }
  
};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "No cipher text given!" << std::endl;
    return 1;
  }

  // First load the text
  std::ifstream ifs(argv[1], std::ios::binary);

  if (!ifs.good()) {
    std::cerr << "Couldn't open book file!" << std::endl;
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

  // we assume the cipher text has been sanitized
  thrust::device_vector<unsigned char> text_clean = text;

  // now we crack the Vigenere cipher
  // first we need to determine the key length
  // use the kappa index of coincidence
  int keyLength = 0;
  {
    bool found = false;
    int shift_idx = 4; // Start at index 4.
    while (!found) {
      // TODO: Use thrust to compute the number of characters that match
      // when shifting text_clean by shift_idx.
      int numMatches = 0; // = ?  TODO
      
      numMatches = thrust::inner_product(text_clean.begin(),
                                         text_clean.end() - shift_idx,
                                         text_clean.begin() + shift_idx,
                                         (int)0,
                                         thrust::plus<int>(),
                                         thrust::equal_to<unsigned char>());


      double ioc = numMatches /
          static_cast<double>((text_clean.size() - shift_idx) / 26.);

      std::cout << "Period " << shift_idx << " ioc: " << ioc << std::endl;
      if (ioc > 1.6) {
        if (keyLength == 0) {
          keyLength = shift_idx;
          shift_idx = 2 * shift_idx - 1; // check double the period to make sure
        } else if (2 * keyLength == shift_idx) {
          found = true;
        } else {
          std::cout << "Unusual pattern in text!" << std::endl;
          exit(1);
        }
      }
      ++shift_idx;
    }
  }

  std::cout << "keyLength: " << keyLength << std::endl;

  // once we know the key length, then we can do frequency analysis on each
  // pos mod length allowing us to easily break each cipher independently
  // you will find the strided_range useful
  // it is located in strided_range_iterator.h and an example
  // of how to use it is located in the that file
  thrust::device_vector<unsigned char> text_copy = text_clean;
  thrust::device_vector<int> dShifts(keyLength);
  typedef thrust::device_vector<unsigned char>::iterator Iterator;

  // TODO: Now that you have determined the length of the key, you need to
  // compute the actual key. To do so, perform keyLength individual frequency
  // analyses on text_copy to find the shift which aligns the most common
  // character in text_copy with the character 'e'. Fill up the
  // dShifts vector with the correct shifts.
  
  for (int i = 0; i < keyLength; i++)
  {
    strided_range<Iterator> it(text_copy.begin() + i, text_copy.end(), keyLength);
    thrust::sort(it.begin(), it.end());
    
    thrust::device_vector<unsigned char> histogram_keys(26);
    thrust::device_vector<int> histogram_values(26);
    
    thrust::reduce_by_key(it.begin(),
                          it.end(),
                          thrust::make_constant_iterator(1),
                          histogram_keys.begin(),
                          histogram_values.begin());
                          
    
    thrust::device_vector<int>::iterator it_max = thrust::max_element(histogram_values.begin(),
                                                                      histogram_values.end());
    
    int max_location = it_max - histogram_values.begin();
    
    dShifts[i] = 4 - max_location;
  }

  std::cout << "\nEncryption key: ";
  for (unsigned int i = 0; i < keyLength; ++i)
    std::cout << static_cast<char>('a' - (dShifts[i] <= 0 ? dShifts[i] :
                                   dShifts[i] - 26));
  std::cout << std::endl;

  // take the shifts and transform cipher text back to plain text
  // TODO : transform the cipher text back to the plain text by using the
  // apply_shift functor.
  
  apply_shift as(thrust::raw_pointer_cast(&dShifts[0]), keyLength);
  
  thrust::transform(text_copy.begin(),
                    text_copy.end(),
                    thrust::make_counting_iterator(0),
                    text_clean.begin(),
                    as);


  thrust::host_vector<unsigned char> h_plain_text = text_clean;

  std::ofstream ofs("plain_text.txt", std::ios::binary);
  ofs.write((char *)&h_plain_text[0], h_plain_text.size());
  ofs.close();

  return 0;
}
