#ifndef __DILATION_H__
#define __DILATION_H__
#include <host_defines.h>

__device__ inline int IMIN(int a, int b)
{
	return a > b ? b : a;
}

__device__ inline int IMAX(int a, int b)
{
	return a > b ? a : b;
}

__device__ inline unsigned char UCMIN(unsigned char a, unsigned char b)
{
	return a > b ? b : a;
}

__device__ inline unsigned char UCMAX(unsigned char a, unsigned char b)
{
	return a > b ? a : b;
}

typedef unsigned char uint8_t;

void NaiveDilation(uint8_t* src, uint8_t* dst, int width, int height, int radio);

void DilationTwoSteps(uint8_t* src, uint8_t* dst, uint8_t* temp, int width, int height, int radio);

void DilationTwoStepsSharedMemory(uint8_t* src, uint8_t* dst, uint8_t* temp, int width, int height, int radio);

void DilationTemplateTwoStepsSharedmemory(uint8_t* src, uint8_t* dst, uint8_t* temp, int width, int height, int radio);

#endif
