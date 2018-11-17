#pragma once
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "device_launch_parameters.h"

#define GRID_SIZE 1024
#define BLOCK_SIZE 1024
#define NUM_INPUT 1000000
#define NUM_OUTPUT 1
#define NUM_KEYS 1

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
	exit(err);
}

struct KeyValuePair {
	char key;
	int value;
};

class KVComparator {
public:
	__host__ __device__ bool operator() (const KeyValuePair &kv1, const KeyValuePair &kv2) {
		unsigned char *temp1 = (unsigned char *) &(kv1.key);
		unsigned char *temp2 = (unsigned char *) &(kv2.key);
		while (*temp1 && *temp2) {
			if (*temp1 == *temp2) {
				temp1++;
				temp2++;
			}
			else {
				if (*temp1 < *temp2) {
					return false;
				}
				else {
					return true;
				}
			}
		}
		return false;
	}
};

__device__ void mapper(char *idata, KeyValuePair *pairs);

__device__ void reducer(KeyValuePair *pairs, int len, int* odata);

__global__ void kernMap(char *idata, KeyValuePair *pairs);

__global__ void kernReduce(KeyValuePair *pairs, int *odata);


