#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <ctime>

#include "erosionMore.h"
#include "Erosion/erosion.h"
#include "Erosion/erosionCPU.h"
#include "Dilation/dilation.h"
#include "Dilation/dilationCPU.h"

const int Width = 320;
const int Height = 256;

inline int CUDADeviceInit(int argc, const char** argv)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaSetDevice(0);

	return 0;
}

void GenerateImage(uint8_t* image, int width, int height)
{
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			image[i * width + j] = static_cast<uint8_t>(rand() % 256);
		}
	}
}

void CheckDiff(uint8_t* himage, uint8_t* dimage, int width, int height)
{
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			if (himage[i * width + j] != dimage[i * width + j])
			{
				std::cout << "Expected: " << static_cast<int>(himage[i * width + j]) << ", actual: " << static_cast<int>(dimage[i * width + j]) << ", on: " << i << ", " << j << std::endl;
				exit(0);
			}
		}
	}
}

void CalculateDilatedImageOnHost(uint8_t* himage_src, uint8_t* himage_dst, int radio)
{
	auto start = std::chrono::system_clock::now();
	dilationCPU(himage_src, himage_dst, Width, Height, radio);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Dilation CPU: " << elapsed_seconds.count() << "s\n";
}

void CalculateDilatedImageOnDeviceUseFilter(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* dimage_tmp, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	DilationFilter(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height , cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared template dilation with a function templated: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnHost(uint8_t* himage_src, uint8_t* himage_dst, int radio)
{
	auto start = std::chrono::system_clock::now();
	erosionCPU(himage_src, himage_dst, Width, Height, radio);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Erosion CPU: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceNaiveErosion(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	NaiveErosion(dimage_src, dimage_dst, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU Naive erosion: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceErosionTwoSteps(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* dimage_tmp, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	ErosionTwoSteps(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps erosion: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceErosionTwoStepsSharedMemory(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* dimage_tmp, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	ErosionTwoStepsSharedMemory(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared erosion: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceErosionTemplateSharedTwoStepsMemory(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* dimage_tmp, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	ErosionTemplateTwoStepsSharedmemory(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared template erosion: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceUseFilter(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* dimage_tmp, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height * sizeof(int), cudaMemcpyHostToDevice);

	ErosionFilter(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height * sizeof(int), cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared memory erosion filter with a function templated: " << elapsed_seconds.count() << "s\n";
}

void CalculateDilatedImageOnDeviceNaiveDilation(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	NaiveDilation(dimage_src, dimage_dst, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU Naive dilation: " << elapsed_seconds.count() << "s\n";
}

void CalculateDilatedImageOnDeviceDilationTwoSteps(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* dimage_tmp, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	DilationTwoSteps(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps dilation: " << elapsed_seconds.count() << "s\n";
}

void CalculateDilatedImageOnDeviceDilationTwoStepsSharedMemory(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* dimage_tmp, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	DilationTwoStepsSharedMemory(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared dilation: " << elapsed_seconds.count() << "s\n";
}

void CalculateDilatedImageOnDeviceDilationTemplateSharedTwoStepsMemory(uint8_t* dimage_src, uint8_t* dimage_dst, uint8_t* dimage_tmp, uint8_t* himage_src, uint8_t* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	DilationTemplateTwoStepsSharedmemory(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared template dilation: " << elapsed_seconds.count() << "s\n";
}

int main(int argc, char* argv[])
{
	CUDADeviceInit(argc, const_cast<const char **>(argv));

	uint8_t* dimage_src;
	uint8_t* dimage_dst;
	uint8_t* dimage_tmp;
	cudaMalloc(&dimage_src, Width * Height);
	cudaMalloc(&dimage_dst, Width * Height);
	cudaMalloc(&dimage_tmp, Width * Height);

	uint8_t* himage_src;
	uint8_t* himage_dst;
	uint8_t* himage_tmp;
	cudaMallocHost(&himage_src, Width * Height);
	cudaMallocHost(&himage_dst, Width * Height);
	cudaMallocHost(&himage_tmp, Width * Height);

	GenerateImage(himage_src, Width, Height);

	for (auto radio = 1; radio <= 15; radio++)
	{
		std::cout << "Radio = " << radio << std::endl;

		// Erosion
		CalculateErodedImageOnHost(himage_src, himage_dst, radio);

		CalculateErodedImageOnDeviceNaiveErosion(dimage_src, dimage_dst, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateErodedImageOnDeviceErosionTwoSteps(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateErodedImageOnDeviceErosionTwoStepsSharedMemory(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateErodedImageOnDeviceErosionTemplateSharedTwoStepsMemory(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateErodedImageOnDeviceUseFilter(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		// Dilation
		CalculateDilatedImageOnHost(himage_src, himage_dst, radio);

		CalculateDilatedImageOnDeviceNaiveDilation(dimage_src, dimage_dst, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateDilatedImageOnDeviceDilationTwoSteps(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateDilatedImageOnDeviceDilationTwoStepsSharedMemory(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateDilatedImageOnDeviceDilationTemplateSharedTwoStepsMemory(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateDilatedImageOnDeviceUseFilter(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);
	}

	std::cout << "Great!!" << std::endl;

	cudaFree(dimage_src);
	cudaFree(dimage_dst);
	cudaFree(dimage_tmp);

	cudaFreeHost(himage_src);
	cudaFreeHost(himage_dst);
	cudaFreeHost(himage_tmp);

	cudaDeviceReset();
	system("Pause");
	return 0;
}
