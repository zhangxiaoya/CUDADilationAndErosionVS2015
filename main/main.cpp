#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <ctime>

#include "erosionFuncTemplate.h"
#include "erosionCPU.h"
#include "erosion.h"
#include <fstream>

const int Width = 320;
const int Height = 256;

inline int cudaDeviceInit(int argc, const char** argv)
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

void populateImage(unsigned char* image, int width, int height)
{
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			image[i * width + j] = static_cast<unsigned char>(rand() % 256);
		}
	}
}

void CheckDiff(unsigned char* himage, unsigned char* dimage, int width, int height)
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

void CalculateDilatedImageOnHost(unsigned char* himage_src, unsigned char* himage_dst, int radio)
{
	auto start = std::chrono::system_clock::now();
	dilationCPU(himage_src, himage_dst, Width, Height, radio);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Dilation CPU: " << elapsed_seconds.count() << "s\n";
}

void CalculateDilatedImageOnDevice(unsigned char* dimage_src, unsigned char* dimage_dst, unsigned char* dimage_tmp, unsigned char* himage_src, unsigned char* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	FilterDilation(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height , cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared template dilation with a function templated: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnHost(unsigned char* himage_src, unsigned char* himage_dst, int radio)
{
	auto start = std::chrono::system_clock::now();
	erosionCPU(himage_src, himage_dst, Width, Height, radio);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Erosion CPU: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceNaiveErosion(unsigned char* dimage_src, unsigned char* dimage_dst, unsigned char* himage_src, unsigned char* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	NaiveErosion(dimage_src, dimage_dst, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU Naive erosion: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceErosionTwoSteps(unsigned char* dimage_src, unsigned char* dimage_dst, unsigned char* dimage_tmp, unsigned char* himage_src, unsigned char* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	ErosionTwoSteps(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps erosion: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceErosionTwoStepsShared(unsigned char* dimage_src, unsigned char* dimage_dst, unsigned char* dimage_tmp, unsigned char* himage_src, unsigned char* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height , cudaMemcpyHostToDevice);

	ErosionTwoStepsShared(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height , cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared erosion: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceErosionTemplateSharedTwoSteps(unsigned char* dimage_src, unsigned char* dimage_dst, unsigned char* dimage_tmp, unsigned char* himage_src, unsigned char* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height, cudaMemcpyHostToDevice);

	ErosionTemplateSharedTwoSteps(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height, cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared template erosion: " << elapsed_seconds.count() << "s\n";
}

void CalculateErodedImageOnDeviceFilter(unsigned char* dimage_src, unsigned char* dimage_dst, unsigned char* dimage_tmp, unsigned char* himage_src, unsigned char* himage_tmp, int radio)
{
	auto start = std::chrono::system_clock::now();
	cudaMemcpy(dimage_src, himage_src, Width * Height * sizeof(int), cudaMemcpyHostToDevice);

	Filter(dimage_src, dimage_dst, dimage_tmp, Width, Height, radio);

	cudaMemcpy(himage_tmp, dimage_dst, Width * Height * sizeof(int), cudaMemcpyDeviceToHost);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU two steps shared template erosion with a function templated: " << elapsed_seconds.count() << "s\n";
}

int main(int argc, char* argv[])
{
	cudaDeviceInit(argc, const_cast<const char **>(argv));

	unsigned char* dimage_src, *dimage_dst, *dimage_tmp;
	unsigned char* himage_src, *himage_dst, *himage_tmp;

	cudaMalloc(&dimage_src, Width * Height );
	cudaMalloc(&dimage_dst, Width * Height );
	cudaMalloc(&dimage_tmp, Width * Height );

	cudaMallocHost(&himage_src, Width * Height);
	cudaMallocHost(&himage_dst, Width * Height);
	cudaMallocHost(&himage_tmp, Width * Height);

	populateImage(himage_src, Width, Height);

	for (auto radio = 1; radio <= 15; radio++)
	{
		std::cout << "Radio = " << radio<<std::endl;

		CalculateErodedImageOnHost(himage_src, himage_dst, radio);

		CalculateErodedImageOnDeviceNaiveErosion(dimage_src, dimage_dst, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateErodedImageOnDeviceErosionTwoSteps(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateErodedImageOnDeviceErosionTwoStepsShared(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateErodedImageOnDeviceErosionTemplateSharedTwoSteps(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateErodedImageOnDeviceFilter(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);

		CalculateDilatedImageOnHost(himage_src, himage_dst, radio);

		CalculateDilatedImageOnDevice(dimage_src, dimage_dst, dimage_tmp, himage_src, himage_tmp, radio);
		CheckDiff(himage_dst, himage_tmp, Width, Height);
	}
//	templateErosionTophatCall(dimage_src, dimage_dst, dimage_tmp, width, height, radio);

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
