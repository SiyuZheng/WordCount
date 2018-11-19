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
#include <sstream>
#include <string>
#include <fstream>
#include <tchar.h>
#include "string"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void cudaMap(char *input, KeyValuePair *pairs) {
	kernMap << < GRID_SIZE, BLOCK_SIZE >> >(input, pairs);
	checkCUDAError("Map kernel failed!");
	cudaDeviceSynchronize();
}

__global__ void kernMap(char *idata, KeyValuePair *pairs) {
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	for (int i = ind; i < NUM_INPUT; i += offset) {
		mapper(&idata[i], &pairs[i * NUM_KEYS]);
	}
}

__device__ void mapper(char *input, KeyValuePair *pairs)
{
	pairs->key = 0;
	char ch = *input;
	if (ch == ' ' || ch == '\n' || ch == ',' || ch == '.')
	{
		pairs->value = 1;
	}
	else
	{
		pairs->value = 0;
	}
	
}


void cudaReduce(KeyValuePair *pairs, int *odata) {
	kernReduce << <GRID_SIZE, BLOCK_SIZE >> >(pairs, odata);
	checkCUDAError("Reduce kernel failed!");
	cudaDeviceSynchronize();
}

__global__ void kernReduce(KeyValuePair *pairs, int *odata) {
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	for (int i = ind; i < NUM_OUTPUT; i += offset) {
		int startIndex = 0;
		int count = 0;
		int valueSize = 0;
		int j;

		for (j = 1; j < NUM_INPUT * NUM_KEYS; j++) {
			if (KVComparator()(pairs[j - 1], pairs[j])) {
				if (count == i) {
					break;
				}
				else {
					count++;
					startIndex = j;
				}
			}
		}
		valueSize = j - startIndex;
		reducer(pairs + startIndex, valueSize, &odata[i]);
	}
}

__device__ void reducer(KeyValuePair *pairs, int len, int* odata)
{
	int sum = 0;
	for (KeyValuePair *pair = pairs; pair != pairs + len; pair++)
	{
		sum += pair->value;
	}
	*odata = sum;
}

void cudaMapReduce(char* input, int *output) {
	char* dev_idata;
	int* dev_odata;
	KeyValuePair *dev_intermediate;

	cudaMalloc(&dev_idata, NUM_INPUT * sizeof(char));
	cudaMalloc(&dev_intermediate, NUM_INPUT * NUM_KEYS * sizeof(KeyValuePair));
	cudaMalloc(&dev_odata, NUM_OUTPUT * sizeof(int));

	cudaMemcpy(dev_idata, input, NUM_INPUT * sizeof(char), cudaMemcpyHostToDevice);

	cudaMap(dev_idata, dev_intermediate);

	thrust::device_ptr<KeyValuePair> dev_ptr(dev_intermediate);
	thrust::sort(dev_ptr, dev_ptr + NUM_INPUT * NUM_KEYS, KVComparator());

	cudaReduce(dev_intermediate, dev_odata);

	cudaMemcpy(output, dev_odata, NUM_OUTPUT * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_idata);
	cudaFree(dev_intermediate);
	cudaFree(dev_odata);
}


int chars = 0, words = 0, lines = 1;
char c;

void CPUCounting(FILE *file)
{
	while ((c = fgetc(file)) != EOF)
	{
		chars++;
		if (c != ' '&& c != ',' && c != '\n')
		{
			words++;
			while ((c = fgetc(file)) != EOF)
			{
				chars++;
				if (c != ' '&& c != ','&& c != '\n')
				{
				}
				else if (c == '\n')
				{

					lines++;
					break;
				}
				else if (c == ' ' || c == ',' || c == '\n')
					break;
				else
				{
					break;
				}
			}
		}

		else if (c == '\n')
		{
			lines++;
		}
	}
}

int main(int argc, char* argv[]) {
	//// read by line
	//std::ifstream infile("hamlet.txt");
	//std::string line;
	//if (infile) {
	//	while (std::getline(infile, line)) {
	//		std::cout << line << std::endl;
	//	}
	//}
	//else {
	//	std::cout << "no such file" << std::endl;
	//}

	char* idata = new char[NUM_INPUT];
	int* odata = new int[NUM_OUTPUT];
	char* filename = "test.txt";
	FILE* fp;
	fp = fopen(filename, "r");
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
		idata[i] = ch;
		printf("%c", ch);
	}
	FILE* fp2;
	fp2 = fopen(filename, "r");
	CPUCounting(fp2);
	if (fp != NULL) {
		fclose(fp);
	}
	if (fp2 != NULL) {
		fclose(fp);
	}
	cudaMapReduce(idata, odata);

	for (int i = 0; i < argc; i++)
	{
		std::cout << "CPU computing: " << std::endl;
		std::cout << "Total word count: " << words << std::endl;
	}
	for (int i = 0; i < NUM_OUTPUT; i++)
	{
		std::cout << "GPU computing: " << std::endl;
		std::cout << "Total word count: " << odata[i] << std::endl;
	}
	delete idata;
	delete odata;
	return 0;
}