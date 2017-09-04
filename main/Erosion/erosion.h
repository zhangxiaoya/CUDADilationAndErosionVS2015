#ifndef __EROSION_H__
#define __EROSION_H__

typedef unsigned char uint8_t;

void NaiveErosion(uint8_t* src, uint8_t* dst, int width, int height, int radio);

void ErosionTwoSteps(uint8_t* src, uint8_t* dst, uint8_t* temp, int width, int height, int radio);

void ErosionTwoStepsSharedMemory(uint8_t* src, uint8_t* dst, uint8_t* temp, int width, int height, int radio);

void ErosionTemplateTwoStepsSharedmemory(uint8_t* src, uint8_t* dst, uint8_t* temp, int width, int height, int radio);

#endif
