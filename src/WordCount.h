#pragma once
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

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

struct keyValueCompare {
	__host__ __device__ bool operator() (const KeyValuePair &lhs, const KeyValuePair &rhs) {
		void *char_lhs = (unsigned char *) &(lhs.key);
		void *char_rhs = (unsigned char *) &(rhs.key);
		for (int i = 0; i < sizeof(char); i++) {
			unsigned char *p1 = (unsigned char *)char_lhs + i;
			unsigned char *p2 = (unsigned char *)char_rhs + i;
			if (*p1 < *p2) {
				return true;
			}
			else if (*p1 > *p2) {
				return false;
			}
		}
		return false;
	}
};

__device__ void mapper(char *input, KeyValuePair *pairs);


__device__ void reducer(KeyValuePair *pairs, int len, int* output);

void cudaMap(char *input, KeyValuePair *pairs);

void cudaReduce(KeyValuePair *pairs, int *output);

__global__ void mapKernel(char *input, KeyValuePair *pairs);

__global__ void reduceKernel(KeyValuePair *pairs, int *output);


