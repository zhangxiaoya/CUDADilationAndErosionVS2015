
#include <helper_cuda.h>       // helper for CUDA Error handling and initialization
#include <host_defines.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "dilation.h"

__global__ void NaiveDilationKernel(unsigned char* src, unsigned char* dst, int width, int height, int radio)
{
	int colCount = blockIdx.x * blockDim.x + threadIdx.x;
	int rowCount = blockIdx.y * blockDim.y + threadIdx.y;

	if (rowCount >= height || colCount >= width)
	{
		return;
	}
	unsigned int startRow = IMAX(rowCount - radio, 0);
	unsigned int endRow = IMIN(height - 1, rowCount + radio);
	unsigned int startCol = IMAX(colCount - radio, 0);
	unsigned int endCol = IMIN(width - 1, colCount + radio);

	unsigned char maxValue = 0;

	for (int r = startRow; r <= endRow; r++)
	{
		for (int c = startCol; c <= endCol; c++)
		{
			maxValue = UCMAX(maxValue, src[r * width + c]);
		}
	}
	dst[rowCount * width + colCount] = maxValue;
}

void NaiveDilation(uint8_t* src, uint8_t* dst, int width, int height, int radio)
{
	dim3 block(32, 32);
	dim3 grid(ceil(static_cast<float>(width) / block.x), ceil(static_cast<float>(height) / block.y));
	NaiveDilationKernel<<<grid, block >>>(src, dst, width, height, radio);
	auto cudaerr = cudaDeviceSynchronize();
}

__global__ void DilationForEachCol(unsigned char* src, unsigned char* dst, int width, int height, int radio)
{
	int colCount = blockIdx.x * blockDim.x + threadIdx.x;
	int rowCount = blockIdx.y * blockDim.y + threadIdx.y;
	if (rowCount >= height || colCount >= width)
	{
		return;
	}
	unsigned int startRow = IMAX(rowCount - radio, 0);
	unsigned int endRow = IMIN(height - 1, rowCount + radio);
	unsigned char maxValue = 0;

	for (int row = startRow; row <= endRow; row++)
	{
		maxValue = IMAX(maxValue, src[row * width + colCount]);
	}
	dst[rowCount * width + colCount] = maxValue;
}

__global__ void DilationForEachRow(unsigned char* src, unsigned char * dst, int width, int height, int radio)
{
	int colCount = blockIdx.x * blockDim.x + threadIdx.x;
	int rowCount = blockIdx.y * blockDim.y + threadIdx.y;
	if (rowCount >= height || colCount >= width)
	{
		return;
	}
	unsigned int startCol = IMAX(colCount - radio, 0);
	unsigned int endCol = IMIN(width - 1, colCount + radio);
	unsigned char maxValue = 0;

	for (int col = startCol; col <= endCol; col++)
	{
		maxValue = IMAX(maxValue, src[rowCount * width + col]);
	}
	dst[rowCount * width + colCount] = maxValue;
}

void DilationTwoSteps(unsigned char * src, unsigned char * dst, unsigned char * temp, int width, int height, int radio)
{
	dim3 block(16, 16);
	dim3 grid(ceil(static_cast<float>(width) / block.x), ceil(static_cast<float>(height) / block.y));
	DilationForEachRow<<<grid, block >>>(src, temp, width, height, radio);
	auto cudaerr = cudaDeviceSynchronize();

	DilationForEachCol<<<grid, block >>>(temp, dst, width, height, radio);
	cudaerr = cudaDeviceSynchronize();
}

__global__ void DilationSharedMemoryForEachCol(unsigned char * src, unsigned char * dst, int radio, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ unsigned char smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	auto x = bx * tile_w + tx;
	auto y = by * tile_h + ty - radio;

	smem[ty * blockDim.x + tx] = 0;
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
	for (auto yy = 1; yy <= 2 * radio; yy++)
	{
		val = UCMAX(val, smem_thread[yy * blockDim.x]);
	}
	dst[y * width + x] = val;
}

__global__ void DilationSharedMemoryForEachRow(unsigned char * src, unsigned char * dst, int radio, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ unsigned char smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	auto x = bx * tile_w + tx - radio;
	auto y = by * tile_h + ty;

	smem[ty * blockDim.x + tx] = 0;
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
	for (auto xx = 1; xx <= 2 * radio; xx++)
	{
		val = UCMAX(val, smem_thread[xx]);
	}
	dst[y * width + x] = val;
}

void DilationTwoStepsSharedMemory(unsigned char * src, unsigned char* dst, unsigned char * temp, int width, int height, int radio)
{
	auto tile_w = 640;
	auto tile_h = 1;
	dim3 block2(tile_w + (2 * radio), tile_h);
	dim3 grid2(ceil(static_cast<float>(width) / tile_w), ceil(static_cast<float>(height) / tile_h));
	DilationSharedMemoryForEachRow<<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, radio, width, height, tile_w, tile_h);
	auto cudaerr = cudaDeviceSynchronize();

	tile_w = 8;
	tile_h = 64;
	dim3 block3(tile_w, tile_h + (2 * radio));
	dim3 grid3(ceil(static_cast<float>(width) / tile_w), ceil(static_cast<float>(height) / tile_h));
	DilationSharedMemoryForEachCol <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, radio, width, height, tile_w, tile_h);
	cudaerr = cudaDeviceSynchronize();
}

template<const int radio> __global__ void DilationTemplateSharedForEachCol(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ unsigned char smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	auto x = bx * tile_w + tx;
	auto y = by * tile_h + ty - radio;
	smem[ty * blockDim.x + tx] = 0;
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
	unsigned char val = smem_thread[0];
#pragma unroll
	for (auto yy = 1; yy <= 2 * radio; yy++)
	{
		val = UCMAX(val, smem_thread[yy * blockDim.x]);
	}
	dst[y * width + x] = val;
}

template<const int radio> __global__ void DilationTemplateSharedForEachRow(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ unsigned char smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	auto x = bx * tile_w + tx - radio;
	auto y = by * tile_h + ty;
	smem[ty * blockDim.x + tx] = 0;
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
	unsigned char val = smem_thread[0];
#pragma unroll
	for (auto xx = 1; xx <= 2 * radio; xx++)
	{
		val = UCMAX(val, smem_thread[xx]);
	}
	dst[y * width + x] = val;
}

void DilationTemplateTwoStepsSharedmemory(unsigned char * src, unsigned char * dst, unsigned char * temp, int width, int height, int radio)
{
	auto tile_w1 = 256, tile_h1 = 1;
	dim3 block2(tile_w1 + (2 * radio), tile_h1);
	dim3 grid2(ceil(static_cast<float>(width) / tile_w1), ceil(static_cast<float>(height) / tile_h1));

	auto tile_w2 = 4, tile_h2 = 64;
	dim3 block3(tile_w2, tile_h2 + (2 * radio));
	dim3 grid3(ceil(static_cast<float>(width) / tile_w2), ceil(static_cast<float>(height) / tile_h2));

	switch (radio) {
	case 1:
		DilationTemplateSharedForEachRow<1> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<1> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 2:
		DilationTemplateSharedForEachRow<2> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<2> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 3:
		DilationTemplateSharedForEachRow<3> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<3> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 4:
		DilationTemplateSharedForEachRow<4> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<4> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 5:
		DilationTemplateSharedForEachRow<5> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<5> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 6:
		DilationTemplateSharedForEachRow<6> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<6> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 7:
		DilationTemplateSharedForEachRow<7> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<7> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 8:
		DilationTemplateSharedForEachRow<8> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<8> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 9:
		DilationTemplateSharedForEachRow<9> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<9> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 10:
		DilationTemplateSharedForEachRow<10> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<10> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 11:
		DilationTemplateSharedForEachRow<11> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<11> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 12:
		DilationTemplateSharedForEachRow<12> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<12> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 13:
		DilationTemplateSharedForEachRow<13> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<13> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 14:
		DilationTemplateSharedForEachRow<14> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<14> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	case 15:
		DilationTemplateSharedForEachRow<15> <<<grid2, block2, block2.y*block2.x * sizeof(int) >>>(src, temp, width, height, tile_w1, tile_h1);
		checkCudaErrors(cudaDeviceSynchronize());
		DilationTemplateSharedForEachCol<15> <<<grid3, block3, block3.y*block3.x * sizeof(int) >>>(temp, dst, width, height, tile_w2, tile_h2);
		break;
	default:
		break;;
	}
	auto cudaerr = cudaDeviceSynchronize();
}