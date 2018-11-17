#include <cstdio>
#include <stdlib.h>
#include <string>
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "WordCount.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "device_launch_parameters.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

__device__ void mapper(char *input, KeyValuePair *pairs) 
{
    // We set the key of each input to 0.
    pairs->key = 0;
    char ch = *input;	
	//pairs->key = ch;
    // We check if the input array has a space or a new line and set the value accordingly.
	//If so this will count the number of words in a file.
    if (ch == ' '||ch == '\n') 
	{ 
		pairs->value = 1;
	}   
	else 
	{
        pairs->value = 0;
    }
}

__device__ void reducer(KeyValuePair *pairs, int len, int* output) 
{
    int wordCount = 0;
    for (KeyValuePair *pair = pairs; pair != pairs + len; pair++) 
	{
        if(pair->value == 1) 
		{
            wordCount++;
        }
    }
    *output = wordCount;
}

void cudaMap(char *input, KeyValuePair *pairs) {
	mapKernel <<< GRID_SIZE, BLOCK_SIZE >>>(input, pairs);
	checkCUDAError("Map kernel failed!");
	cudaDeviceSynchronize();
}

void cudaReduce(KeyValuePair *pairs, int *output) {
	reduceKernel << <GRID_SIZE, BLOCK_SIZE >> >(pairs, output);
	checkCUDAError("Reduce kernel failed!");
	cudaDeviceSynchronize();
}

__global__ void mapKernel(char *input, KeyValuePair *pairs) {
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
		i < NUM_INPUT;
		i += blockDim.x * gridDim.x) {
		mapper(&input[i], &pairs[i * NUM_KEYS]);
	}
}

__global__ void reduceKernel(KeyValuePair *pairs, int *output) {
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
		i < NUM_OUTPUT;
		i += blockDim.x * gridDim.x) {
		int startIndex = 0;
		int count = 0;
		int valueSize = 0;
		int j;

		for (j = 1; j < NUM_INPUT * NUM_KEYS; j++) {
			if (keyValueCompare()(pairs[j - 1], pairs[j])) {
				if (count == i) {
					// This thread has found the correct number
					// There is a bit of warp divergence here as some threads
					// break before others, but we still make the most out of it
					// by calling the reducer at the very end, so there is not
					// any warp divergence where the bulk of the computation
					// should occur (the reducer).
					break;
				}
				else {
					count++;
					startIndex = j;
				}
			}
		}

		if (count < i) {
			// This thread doesn't need to process a key. We won't get here, but
			// this code is just there for assurance.
			return;
		}

		valueSize = j - startIndex;

		// Run the reducer
		reducer(pairs + startIndex, valueSize, &output[i]);
	}
}

void cudaMapReduce(char* input, int *output) {
	// Create device pointers
	char* dev_input;
	int* dev_output;
	KeyValuePair *dev_pairs;

	// Determine sizes in bytes
	size_t input_size = NUM_INPUT * sizeof(char);
	size_t output_size = NUM_OUTPUT * sizeof(int);
	size_t pairs_size = NUM_INPUT * NUM_KEYS * sizeof(KeyValuePair);

	// Initialize device memory (we can utilize more space by waiting to
	// initialize the output array until we're done with the input array)
	cudaMalloc(&dev_input, input_size);
	cudaMalloc(&dev_pairs, pairs_size);

	 //Copy input data over
	cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);
	//cudaMemset(dev_pairs, 0, pairs_size);

	// Run the mapper kernel
	cudaMap(dev_input, dev_pairs);

	// Convert the pointer to device memory for the key/value pairs that is
	// recognizable by the cuda thrust library
	thrust::device_ptr<KeyValuePair> dev_ptr(dev_pairs);

	// Sort the key/value pairs. By using the thrust library, we don't have to
	// write this code ourselves, and it's already optimized for parallel
	// computation
	thrust::sort(dev_ptr, dev_ptr + NUM_INPUT * NUM_KEYS, keyValueCompare());

	// Free GPU space for the input
	cudaFree(dev_input);
	// Allocate GPU space for the output
	cudaMalloc(&dev_output, output_size);

	// Run the reducer kernel
	cudaReduce(dev_pairs, dev_output);

	// Allocate space on the host for the output array and copy the data to it
	cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);

	// Free GPU memory for the key/value pairs and output array
	cudaFree(dev_pairs);
	cudaFree(dev_output);
}

int main(int argc, char* argv[]) {
	size_t input_size = NUM_INPUT * sizeof(char);
	size_t output_size = NUM_OUTPUT * sizeof(int);
	char *input = (char *)malloc(input_size);
	int *output = (int *)malloc(output_size);
	FILE* fp;
	fp = fopen("test.txt", "r");
	int i = 0;
	int ch;
	while (1) {
		if (fp == NULL) {
			printf("File didn't open");
			break;
		}
		ch = fgetc(fp);
		if (ch == EOF) {
			break;
		}
		i++;

		input[i] = ch;
		printf("%c", ch);
	}
	if (fp != NULL) {
		fclose(fp);
	}
	cudaMapReduce(input, output);
	for (size_t i = 0; i < NUM_OUTPUT; i++)
	{
		printf("The total number of words in the file are: %d\n", output[i]);
	}
	delete input;
	delete output;
	return 0;
}