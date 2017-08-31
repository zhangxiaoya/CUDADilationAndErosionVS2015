
#include <helper_cuda.h>       // helper for CUDA Error handling and initialization
#include <host_defines.h>

__device__ int IMIN(int a, int b)
{
	return a > b ? b : a;
}

__device__ int IMAX(int a, int b)
{
	return a > b ? a : b;
}

__device__ unsigned char UCMIN(unsigned char a, unsigned char b)
{
	return a > b ? b : a;
}

__device__ unsigned char UCMAX(unsigned char a, unsigned char b)
{
	return a > b ? a : b;
}

/**
 * Naive erosion kernel with each thread processing a square area.
 */
__global__ void NaiveErosionKernel(unsigned char* src, unsigned char* dst, int width, int height, int radio)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= height || x >= width)
	{
		return;
	}
	unsigned int start_i = IMAX(y - radio, 0);
	unsigned int end_i = IMIN(height - 1, y + radio);
	unsigned int start_j = IMAX(x - radio, 0);
	unsigned int end_j = IMIN(width - 1, x + radio);

	unsigned char value = 255;

	for (int i = start_i; i <= end_i; i++)
	{
		for (int j = start_j; j <= end_j; j++)
		{
			value = UCMIN(value, src[i * width + j]);
		}
	}
	dst[y * width + x] = value;
}

void NaiveErosion(unsigned char* src, unsigned char* dst, int width, int height, int radio)
{
	dim3 block(32, 32);
	dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y));
	NaiveErosionKernel<<<grid,block>>>(src, dst, width, height, radio);
	cudaError_t cudaerr = cudaDeviceSynchronize();
}

/**
 * Two steps erosion using separable filters
 */
__global__ void ErosionStep2(unsigned char* src, unsigned char* dst, int width, int height, int radio)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width)
	{
        return;
    }
    unsigned int start_i = max(y - radio, 0);
    unsigned int end_i = min(height - 1, y + radio);
    unsigned char value = 255;
    for (int i = start_i; i <= end_i; i++)
	{
        value = min(value, src[i * width + x]);
    }
    dst[y * width + x] = value;
}

__global__ void ErosionStep1(unsigned char* src, unsigned char * dst, int width, int height, int radio)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width)
	{
        return;
    }
    unsigned int start_j = max(x - radio, 0);
    unsigned int end_j = min(width - 1, x + radio);
    unsigned char value = 255;
    for (int j = start_j; j <= end_j; j++)
	{
        value = min(value, src[y * width + j]);
    }
    dst[y * width + x] = value;
}

void ErosionTwoSteps(unsigned char * src, unsigned char * dst, unsigned char * temp, int width, int height, int radio)
{
    dim3 block(16, 16);
    dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y));
    ErosionStep1<<<grid,block>>>(src, temp, width, height, radio);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    ErosionStep2<<<grid,block>>>(temp, dst, width, height, radio);
    cudaerr = cudaDeviceSynchronize();
}


/**
 * Two steps erosion using separable filters with shared memory.
 */
__global__ void ErosionSharedStep2(unsigned char * src, unsigned char *src_src, unsigned char * dst, int radio, int width, int height, int tile_w, int tile_h)
{
    extern __shared__ unsigned char smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx;
    int y = by * tile_h + ty - radio;
    smem[ty * blockDim.x + tx] = 255;
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
    unsigned char * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
    unsigned char val = smem_thread[0];
    for (int yy = 1; yy <= 2 * radio; yy++)
	{
        val = min(val, smem_thread[yy * blockDim.x]);
    }
    dst[y * width + x] = val;
}

__global__ void ErosionSharedStep1(unsigned char * src, unsigned char * dst, int radio, int width, int height, int tile_w, int tile_h)
{
    extern __shared__ unsigned char smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx - radio;
    int y = by * tile_h + ty;
    smem[ty * blockDim.x + tx] = 255;
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
    unsigned char* smem_thread = &smem[ty * blockDim.x + tx - radio];
    unsigned char val = smem_thread[0];
    for (int xx = 1; xx <= 2 * radio; xx++)
	{
        val = min(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}

void ErosionTwoStepsShared(unsigned char * src, unsigned char* dst, unsigned char * temp, int width, int height, int radio)
{
    int tile_w = 640;
    int tile_h = 1;
    dim3 block2(tile_w + (2 * radio), tile_h);
    dim3 grid2(ceil((float)width / tile_w), ceil((float)height / tile_h));
    ErosionSharedStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, radio, width, height, tile_w, tile_h);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    tile_w = 8;
    tile_h = 64;
    dim3 block3(tile_w, tile_h + (2 * radio));
    dim3 grid3(ceil((float)width / tile_w), ceil((float)height / tile_h));
    ErosionSharedStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, src, dst, radio, width, height, tile_w, tile_h);
    cudaerr = cudaDeviceSynchronize();
}

template<const int radio> __global__ void ErosionTemplateSharedStep2(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h)
{
    extern __shared__ unsigned char smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx;
    int y = by * tile_h + ty - radio;
    smem[ty * blockDim.x + tx] = 255;
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
    unsigned char * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
    unsigned char val = smem_thread[0];
#pragma unroll
    for (int yy = 1; yy <= 2 * radio; yy++)
	{
        val = min(val, smem_thread[yy * blockDim.x]);
    }
    dst[y * width + x] = val;
}

template<const int radio> __global__ void ErosionTemplateSharedStep1(unsigned char * src, unsigned char * dst, int width, int height, int tile_w, int tile_h)
{
    extern __shared__ unsigned char smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx - radio;
    int y = by * tile_h + ty;
    smem[ty * blockDim.x + tx] = 255;
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
    unsigned char * smem_thread = &smem[ty * blockDim.x + tx - radio];
	unsigned char val = smem_thread[0];
#pragma unroll
    for (int xx = 1; xx <= 2 * radio; xx++)
	{
        val = min(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}

void ErosionTemplateSharedTwoSteps(unsigned char * src, unsigned char * dst, unsigned char * temp, int width, int height, int radio)
{
    int tile_w1 = 256, tile_h1 = 1;
    dim3 block2(tile_w1 + (2 * radio), tile_h1);
    dim3 grid2(ceil((float)width / tile_w1), ceil((float)height / tile_h1));
    int tile_w2 = 4, tile_h2 = 64;
    dim3 block3(tile_w2, tile_h2 + (2 * radio));
    dim3 grid3(ceil((float)width / tile_w2), ceil((float)height / tile_h2));
    switch (radio) {
        case 1:
            ErosionTemplateSharedStep1<1><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<1><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 2:
            ErosionTemplateSharedStep1<2><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<2><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 3:
            ErosionTemplateSharedStep1<3><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<3><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 4:
            ErosionTemplateSharedStep1<4><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<4><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 5:
            ErosionTemplateSharedStep1<5><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<5><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 6:
            ErosionTemplateSharedStep1<6><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<6><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 7:
            ErosionTemplateSharedStep1<7><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<7><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 8:
            ErosionTemplateSharedStep1<8><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<8><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 9:
            ErosionTemplateSharedStep1<9><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<9><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 10:
            ErosionTemplateSharedStep1<10><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<10><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 11:
            ErosionTemplateSharedStep1<11><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<11><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 12:
            ErosionTemplateSharedStep1<12><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<12><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 13:
            ErosionTemplateSharedStep1<13><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<13><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 14:
            ErosionTemplateSharedStep1<14><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<14><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 15:
            ErosionTemplateSharedStep1<15><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            checkCudaErrors(cudaDeviceSynchronize());
            ErosionTemplateSharedStep2<15><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
    }
    cudaError_t cudaerr = cudaDeviceSynchronize();
}