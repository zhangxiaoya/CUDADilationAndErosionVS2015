#pragma once
void Filter(int* src, int* dst, int* temp, int width, int height, int radio);

void FilterDilation(unsigned char* src, unsigned char* dst, unsigned char* temp, int width, int height, int radio);
