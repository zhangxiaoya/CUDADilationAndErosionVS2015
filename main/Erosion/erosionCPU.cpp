#include <math.h>
#include <limits>

#include "erosionCPU.h"
#include "../Util/Util.h"

void erosionCPU(unsigned char* src, unsigned char* dst, int width, int height, int radio)
{
	auto tmp = new unsigned char[width * height];
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			int start_j = imax(0, j - radio);
			int end_j = imin(width - 1, j + radio);
			auto value = std::numeric_limits<unsigned char>::max();
			for (auto jj = start_j; jj <= end_j; jj++)
			{
				value = ucMin(src[i * width + jj], value);
			}
			tmp[i * width + j] = value;
		}
	}
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			int start_i = imax(0, i - radio);
			int end_i = imin(height - 1, i + radio);
			auto value = std::numeric_limits<unsigned char>::max();
			for (auto ii = start_i; ii <= end_i; ii++)
			{
				value = ucMin(tmp[ii * width + j], value);
			}
			dst[i * width + j] = value;
		}
	}
	delete[] tmp;
}
