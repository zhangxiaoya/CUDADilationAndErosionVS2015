#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

typedef unsigned char(*pointFunction_t)(unsigned char, unsigned char);

__device__ unsigned char pComputeMin(unsigned char a, unsigned char b)
{
	return (a < b) ? a : b;
}

__device__
unsigned char pComputeMax(unsigned char a, unsigned char b)
{
	return (a > b) ? a : b;
}

template<const unsigned char boundaryValue>
__device__ void FilterStep2K(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h, const int radio, const pointFunction_t pPointOperation)
{
    extern __shared__ unsigned char smem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

	auto x = bx * tile_w + tx;
	auto y = by * tile_h + ty - radio;

    smem[ty * blockDim.x + tx] = boundaryValue;
    __syncthreads();
    if (x >= width || y < 0 || y >= height)
	{
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (y < (by * tile_h) || y >= ((by + 1) * tile_h))
	{
        return;
    }
	auto smem_thread = &smem[(ty - radio) * blockDim.x + tx];
	auto val = smem_thread[0];
#pragma unroll
    for (auto yy = 1; yy <= 2 * radio; yy++)
	{
        val = pPointOperation(val, smem_thread[yy * blockDim.x]);
    }
    dst[y * width + x] = val;
}

template<const unsigned char boundaryValue>
__device__ void FilterStep1K(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h, const int radio, const pointFunction_t pPointOperation)
{
    extern __shared__ unsigned char smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
	auto x = bx * tile_w + tx - radio;
	auto y = by * tile_h + ty;
    smem[ty * blockDim.x + tx] = boundaryValue;
    __syncthreads();
    if (x < 0 || x >= width || y >= height)
	{
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w))
	{
        return;
    }
	auto smem_thread = &smem[ty * blockDim.x + tx - radio];
	auto val = smem_thread[0];
#pragma unroll
    for (auto xx = 1; xx <= 2 * radio; xx++)
	{
        val = pPointOperation(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}

__global__ void FilterStep1(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h, const int radio)
{
    FilterStep1K<255>(src, dst, width, height, tile_w, tile_h, radio, pComputeMin);
}

__global__ void FilterStep2(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h, const int radio)
{
    FilterStep2K<255>(src, dst, width, height, tile_w, tile_h, radio, pComputeMin);
}

void Filter(unsigned char* src, unsigned char* dst, unsigned char* temp, int width, int height, int radio)
{
	// the host-side function pointer to your __device__ function
	// pointFunction_t h_pointFunction;

	// in host code: copy the function pointers to their host equivalent
	// cudaMemcpyFromSymbol(&h_pointFunction, pComputeMin, sizeof(pointFunction_t));

	auto tile_w1 = 256, tile_h1 = 1;
	dim3 block2(tile_w1 + (2 * radio), tile_h1);
	dim3 grid2(ceil(static_cast<float>(width) / tile_w1), ceil(static_cast<float>(height) / tile_h1));
	auto tile_w2 = 4, tile_h2 = 64;
	dim3 block3(tile_w2, tile_h2 + (2 * radio));
	dim3 grid3(ceil(static_cast<float>(width) / tile_w2), ceil(static_cast<float>(height) / tile_h2));
	FilterStep1<<<grid2,block2,block2.y * block2.x * sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, radio);
	(cudaDeviceSynchronize());
	FilterStep2<<<grid3,block3,block3.y * block3.x * sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, radio);
	auto cudaerr = cudaDeviceSynchronize();
}

__global__ void FilterDStep1(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h, const int radio)
{
    FilterStep1K<0>(src, dst, width, height, tile_w, tile_h, radio, pComputeMax);
}

__global__ void FilterDStep2(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h, const int radio)
{
    FilterStep2K<0>(src, dst, width, height, tile_w, tile_h, radio, pComputeMax);
}

void FilterDilation(unsigned char* src, unsigned char* dst, unsigned char* temp, int width, int height, int radio)
{
	// //the host-side function pointer to your __device__ function
	// pointFunction_t h_pointFunction;

	// //in host code: copy the function pointers to their host equivalent
	// cudaMemcpyFromSymbol(&h_pointFunction, pComputeMin, sizeof(pointFunction_t));

	auto tile_w1 = 256;
	auto tile_h1 = 1;

	dim3 block2(tile_w1 + (2 * radio), tile_h1);
	dim3 grid2(ceil(static_cast<float>(width) / tile_w1), ceil(static_cast<float>(height) / tile_h1));

	auto tile_w2 = 4;
	auto tile_h2 = 64;

	dim3 block3(tile_w2, tile_h2 + (2 * radio));
	dim3 grid3(ceil(static_cast<float>(width) / tile_w2), ceil(static_cast<float>(height) / tile_h2));

	FilterDStep1<<<grid2,block2,block2.y * block2.x>>>(src, temp, width, height, tile_w1, tile_h1, radio);
	(cudaDeviceSynchronize());
	FilterDStep2<<<grid3,block3,block3.y * block3.x>>>(temp, dst, width, height, tile_w2, tile_h2, radio);

	auto cudaerr = cudaDeviceSynchronize();
}
