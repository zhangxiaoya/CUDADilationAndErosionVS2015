#ifndef __DILATION_H__
#define __DILATION_H__

void NaiveDilation(uint8_t* src, uint8_t* dst, int width, int height, int radio);

void DilationTwoSteps(uint8_t* src, uint8_t* dst, uint8_t* temp, int width, int height, int radio);

void DilationTwoStepsSharedMemory(uint8_t* src, uint8_t* dst, uint8_t* temp, int width, int height, int radio);

void DilationTemplateTwoStepsSharedmemory(uint8_t* src, uint8_t* dst, uint8_t* temp, int width, int height, int radio);

#endif
