#ifndef __UTIL_H__
#define __UTIL_H__
#include <host_defines.h>

typedef unsigned char uint8_t;

__device__ inline int IMIN(int a, int b)
{
	return a > b ? b : a;
}

__device__ inline int IMAX(int a, int b)
{
	return a > b ? a : b;
}

__device__ inline unsigned char UCMAX(unsigned char a, unsigned char b)
{
	return a > b ? a : b;
}

__device__ inline unsigned char UCMIN(unsigned char a, unsigned char b)
{
	return a > b ? b : a;
}
#define imax(a,b) (a > b) ? a : b;
#define imin(a,b) (a < b) ? a : b;

inline unsigned char ucMax(uint8_t a, uint8_t b)
{
	return (a > b) ? a : b;
}

inline unsigned char ucMin(uint8_t a, uint8_t b)
{
	return (a < b) ? a : b;
}

#endif
