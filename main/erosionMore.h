#ifndef __EROSION_MORE_H__
#define __EROSION_MORE_H__

void ErosionFilter(unsigned char* src, unsigned char* dst, unsigned char* temp, int width, int height, int radio);

void DilationFilter(unsigned char* src, unsigned char* dst, unsigned char* temp, int width, int height, int radio);

#endif
