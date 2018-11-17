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
	if (ch == ' ' || ch == '\n')
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

const int N = 20;
const int blocksize = 20;

__global__
void hello(char *a, int *c, int size)
{
	int i = 0, k = 0;
	int count = 1;

	if (threadIdx.x < size) {

		for (i = 0; i < size; i++) {
			if (threadIdx.x != i) {
				for (k = 0; k < N; k++) {
					if (a[N * threadIdx.x + k] != a[N * i + k]) {
						break;
					}
					if (k == N - 1) {
						count++;
					}
				}
			}
		}

		c[threadIdx.x] = count;
	}
	else {
		c[threadIdx.x] = -1;
	}
}

int main(int argc, char* argv[]) {
	char* idata = new char[NUM_INPUT];
	int* odata = new int[NUM_OUTPUT];
	FILE* fp;
	fp = fopen("hamlet.txt", "r");
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
	if (fp != NULL) {
		fclose(fp);
	}
	cudaMapReduce(idata, odata);
	for (int i = 0; i < NUM_OUTPUT; i++)
	{
		printf("The total number of words in the file are: %d\n", odata[i]);
	}
	delete idata;
	delete odata;
	return 0;
	//char words[N][N] =
	//{
	//	"ABCDE",
	//	"xyz",
	//	"Hi",
	//	"japan",
	//	"xyz",
	//	"Hi",
	//	"cup",
	//	"paper",
	//	"Hi",
	//	"Apple"
	//};

	//int size = 10;

	//int count[N];

	//char answer_words[N][N];
	//int answer_count[N];

	//char *ad;
	//int *cd;

	//const int csize = N*N * sizeof(char);
	//const int isize = N * sizeof(int);

	//cudaMalloc((void**)&ad, csize);
	//cudaMalloc((void**)&cd, isize);

	//cudaMemcpy(ad, words, csize, cudaMemcpyHostToDevice);

	//dim3 dimBlock(blocksize, 1);
	//dim3 dimGrid(1, 1);
	//hello << <dimGrid, dimBlock >> >(ad, cd, size);
	//cudaMemcpy(count, cd, isize, cudaMemcpyDeviceToHost);
	//cudaFree(ad);
	//cudaFree(cd);

	//int i = 0, k = 0;

	//int num = 0;
	//int dismatchflag = 0;

	//for (i = 0; i < N; i++) {
	//	if (count[i] == -1) {
	//		break;
	//	}

	//	if (count[i] == 1) {
	//		strcpy(answer_words[num], words[i]);
	//		answer_count[num] = count[i];
	//		num++;
	//	}
	//	else if (count[i] > 1) {
	//		for (k = 0; k < num; k++) {
	//			if (strcmp(words[i], answer_words[k]) == 0) {
	//				dismatchflag = 1;
	//				break;
	//			}
	//		}
	//		if (dismatchflag == 0) {
	//			strcpy(answer_words[num], words[i]);
	//			answer_count[num] = count[i];
	//			num++;
	//		}
	//		else {
	//			dismatchflag = 0;
	//		}
	//	}
	//}

	//for (i = 0; i < num; i++) {
	//	printf("%s, %d\n", answer_words[i], answer_count[i]);
	//}

	//return EXIT_SUCCESS;
}