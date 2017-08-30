typedef int(*pointFunction_t)(int, int);

__device__ inline int pComputeMin(int a, int b)
{
	return (a < b) ? a : b;
}

__device__ inline int pComputeMax(int a, int b)
{
	return (a > b) ? a : b;
}

__device__ void FilterStep2K(int * src, int * dst, int width, int height, int tile_w, int tile_h, const int radio, const pointFunction_t pPointOperation)
{
    extern __shared__ int smem[];

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
    int * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
    int val = smem_thread[0];
#pragma unroll
    for (int yy = 1; yy <= 2 * radio; yy++)
	{
        val = pPointOperation(val, smem_thread[yy * blockDim.x]);
    }
    dst[y * width + x] = val;
}

__device__ void FilterStep1K(int * src, int * dst, int width, int height, int tile_w, int tile_h, const int radio, const pointFunction_t pPointOperation)
{
    extern __shared__ int smem[];
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
    int * smem_thread = &smem[ty * blockDim.x + tx - radio];
    int val = smem_thread[0];
#pragma unroll
    for (int xx = 1; xx <= 2 * radio; xx++)
	{
        val = pPointOperation(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}

__global__ void FilterStep1(int * src, int * dst, int width, int height, int tile_w, int tile_h, const int radio)
{
    FilterStep1K(src, dst, width, height, tile_w, tile_h, radio, pComputeMin);
}

__global__ void FilterStep2(int * src, int * dst, int width, int height, int tile_w, int tile_h, const int radio)
{
    FilterStep2K(src, dst, width, height, tile_w, tile_h, radio, pComputeMin);
}

void Filter(int * src, int * dst, int * temp, int width, int height, int radio)
{
    // //the host-side function pointer to your __device__ function
    // pointFunction_t h_pointFunction;

    // //in host code: copy the function pointers to their host equivalent
    // cudaMemcpyFromSymbol(&h_pointFunction, pComputeMin, sizeof(pointFunction_t));

    int tile_w1 = 256, tile_h1 = 1;
    dim3 block2(tile_w1 + (2 * radio), tile_h1);
    dim3 grid2(ceil((float)width / tile_w1), ceil((float)height / tile_h1));
    int tile_w2 = 4, tile_h2 = 64;
    dim3 block3(tile_w2, tile_h2 + (2 * radio));
    dim3 grid3(ceil((float)width / tile_w2), ceil((float)height / tile_h2));
    switch (radio)
	{
        case 1:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 1);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 1);
            break;
        case 2:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 2);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 2);
            break;
        case 3:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 3);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 3);
            break;
        case 4:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 4);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 4);
            break;
        case 5:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 5);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 5);
            break;
        case 6:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 6);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 6);
            break;
        case 7:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 7);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 7);
            break;
        case 8:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 8);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 8);
            break;
        case 9:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 9);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 9);
            break;
        case 10:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 10);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 10);
            break;
        case 11:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 11);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 11);
            break;
        case 12:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 12);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 12);
            break;
        case 13:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 13);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 13);
            break;
        case 14:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 14);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 14);
            break;
        case 15:
            FilterStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 15);
            (cudaDeviceSynchronize());
            FilterStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 15);
            break;
    }
    cudaError_t cudaerr = cudaDeviceSynchronize();
}

__global__ void FilterDStep1(int * src, int * dst, int width, int height, int tile_w, int tile_h, const int radio)
{
    FilterStep1K(src, dst, width, height, tile_w, tile_h, radio, pComputeMax);
}

__global__ void FilterDStep2(int * src, int * dst, int width, int height, int tile_w, int tile_h, const int radio)
{
    FilterStep2K(src, dst, width, height, tile_w, tile_h, radio, pComputeMax);
}

void FilterDilation(int * src, int * dst, int * temp, int width, int height, int radio)
{
    // //the host-side function pointer to your __device__ function
    // pointFunction_t h_pointFunction;

    // //in host code: copy the function pointers to their host equivalent
    // cudaMemcpyFromSymbol(&h_pointFunction, pComputeMin, sizeof(pointFunction_t));

	int tile_w1 = 256;
	int tile_h1 = 1;

	dim3 block2(tile_w1 + (2 * radio), tile_h1);
    dim3 grid2(ceil((float)width / tile_w1), ceil((float)height / tile_h1));

	int tile_w2 = 4;
	int tile_h2 = 64;

	dim3 block3(tile_w2, tile_h2 + (2 * radio));
    dim3 grid3(ceil((float)width / tile_w2), ceil((float)height / tile_h2));

    switch (radio)
	{
        case 1:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 1);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 1);
            break;
        case 2:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 2);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 2);
            break;
        case 3:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 3);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 3);
            break;
        case 4:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 4);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 4);
            break;
        case 5:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 5);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 5);
            break;
        case 6:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 6);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 6);
            break;
        case 7:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 7);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 7);
            break;
        case 8:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 8);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 8);
            break;
        case 9:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 9);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 9);
            break;
        case 10:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 10);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 10);
            break;
        case 11:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 11);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 11);
            break;
        case 12:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 12);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 12);
            break;
        case 13:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 13);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 13);
            break;
        case 14:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 14);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 14);
            break;
        case 15:
            FilterDStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1, 15);
            (cudaDeviceSynchronize());
            FilterDStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2, 15);
            break;
    }
    cudaError_t cudaerr = cudaDeviceSynchronize();
}