#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <ctime>

#include "erosionFuncTemplate.h"
#include "erosionCPU.h"
#include "erosion.h"
#include <fstream>

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

void populateImage(int* image, int width, int height)
{
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			image[i * width + j] = rand() % 256;
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

inline void ChangeRows(unsigned& row, unsigned& col)
{
	col++;
	if (col == 320)
	{
		col = 0;
		row++;
	}
}

void newpopulateImage(unsigned char* himage, int width, int height)
{
	std::string fileName = "C:\\D\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1.bin";

	auto fin = std::ifstream(fileName, std::fstream::binary | std::fstream::in);

	auto originalPerFramePixelArray = new unsigned char[width * height];

	if (fin.is_open())
	{
		// init space on host

		// init some variables
		unsigned row = 0; // current row index
		unsigned col = 0; // current col index
		auto byteIndex = 2; // current byte index
		auto pixelIndex = 0; // current pixel index

		uint8_t highPart = fin.get();
		uint8_t lowPart = fin.get();

		// main loop to read and load binary file per frame
		// check if is the end of binary file

		// per frame
		while (byteIndex - 2 < width * height * 2)
		{
			// take 16-bit space per pixel

			// but we only need only low part of one pixel (temparory)
			originalPerFramePixelArray[pixelIndex] = lowPart;

			// update these variables
			ChangeRows(row, col);
			highPart = fin.get();
			lowPart = fin.get();
			byteIndex += 2;
			pixelIndex++;
		}

//		if (originalPerFramePixelArray != nullptr)
//		{
//			delete[] originalPerFramePixelArray;
//			originalPerFramePixelArray = nullptr;
//		}
		fin.close();
	}

	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			himage[i * width + j] = originalPerFramePixelArray[i * width + j];
		}
	}
}

int main(int argc, char* argv[])
{
	cudaDeviceInit(argc, const_cast<const char **>(argv));

	unsigned char* dimage_src, *dimage_dst, *dimage_tmp;
	unsigned char* himage_src, *himage_dst, *himage_tmp;

	auto width = 320;
	auto height = 256;
	auto radio = 1;

	cudaMalloc(&dimage_src, width * height );
	cudaMalloc(&dimage_dst, width * height );
	cudaMalloc(&dimage_tmp, width * height );
	cudaMallocHost(&himage_src, width * height);
	cudaMallocHost(&himage_dst, width * height);
	cudaMallocHost(&himage_tmp, width * height);

	// Randomly populate the image
	//	populateImage(himage_src, width, height);
	newpopulateImage(himage_src, width, height);

	for (radio = 1; radio <= 15; radio++)
	{
//		auto start = std::chrono::system_clock::now();
//		 Calculate the eroded image on the host
//		erosionCPU(himage_src, himage_dst, width, height, radio);
//		auto end = std::chrono::system_clock::now();

		auto start = std::chrono::system_clock::now();
		// Calculate the eroded image on the host
		dilationCPU(himage_src, himage_dst, width, height, radio);
		auto end = std::chrono::system_clock::now();

//		std::chrono::duration<double> elapsed_seconds = end - start;
//		std::cout << "Erosion CPU: " << elapsed_seconds.count() << "s\n";

		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "Dilation CPU: " << elapsed_seconds.count() << "s\n";

//		start = std::chrono::system_clock::now();
		// Copy the image from the host to the GPU
//		cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice);
		// Calculate the eroded image on the GPU
//		NaiveErosion(dimage_src, dimage_dst, width, height, radio);
		// Copy the eroded image to the host
//		cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost);
//		end = std::chrono::system_clock::now();

//		elapsed_seconds = end - start;
//		std::cout << "GPU Naive erosion: " << elapsed_seconds.count() << "s\n";
		// Diff the images
//		CheckDiff(himage_dst, himage_tmp, width, height);

//		start = std::chrono::system_clock::now();
		// Copy the image from the host to the GPU
//		cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice);
//		ErosionTwoSteps(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
		// Copy the eroded image to the host
//		cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost);
//		end = std::chrono::system_clock::now();

//		elapsed_seconds = end - start;
//		std::cout << "GPU two steps erosion: " << elapsed_seconds.count() << "s\n";
		// Diff the images
//		CheckDiff(himage_dst, himage_tmp, width, height);

//		start = std::chrono::system_clock::now();
		// Copy the image from the host to the GPU
//		cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice);
//		ErosionTwoStepsShared(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
		// Copy the eroded image to the host
//		cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost);
//		end = std::chrono::system_clock::now();

//		elapsed_seconds = end - start;
//		std::cout << "GPU two steps shared erosion: " << elapsed_seconds.count() << "s\n";
		// Diff the images
//		CheckDiff(himage_dst, himage_tmp, width, height);

//		start = std::chrono::system_clock::now();
		// Copy the image from the host to the GPU
//		cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice);
//		ErosionTemplateSharedTwoSteps(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
		// Copy the eroded image to the host
//		cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost);
//		end = std::chrono::system_clock::now();

//		elapsed_seconds = end - start;
//		std::cout << "GPU two steps shared template erosion: " << elapsed_seconds.count() << "s\n";
		// Diff the images
//		CheckDiff(himage_dst, himage_tmp, width, height);

//		start = std::chrono::system_clock::now();
		// Copy the image from the host to the GPU
//		cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice);
//		Filter(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
		// Copy the eroded image to the host
//		cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost);
//		end = std::chrono::system_clock::now();

//		elapsed_seconds = end - start;
//		std::cout << "GPU two steps shared template erosion with a function templated: " << elapsed_seconds.count() << "s\n";
//		 Diff the images
//		CheckDiff(himage_dst, himage_tmp, width, height);

		start = std::chrono::system_clock::now();
		// Copy the image from the host to the GPU
		cudaMemcpy(dimage_src, himage_src, width * height , cudaMemcpyHostToDevice);
		FilterDilation(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
		// Copy the eroded image to the host
		cudaMemcpy(himage_tmp, dimage_dst, width * height , cudaMemcpyDeviceToHost);
		end = std::chrono::system_clock::now();

		elapsed_seconds = end - start;
		std::cout << "GPU two steps shared template dilation with a function templated: " << elapsed_seconds.count() << "s\n";
		CheckDiff(himage_dst, himage_tmp, width, height);
	}
	//templateErosionTophatCall(dimage_src, dimage_dst, dimage_tmp, width, height, radio);

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
