#pragma once
void NaiveErosion(unsigned char* src, unsigned char* dst, int width, int height, int radio);

void ErosionTwoSteps(unsigned char* src, unsigned char* dst, unsigned char* temp, int width, int height, int radio);

void ErosionTwoStepsShared(unsigned char* src, unsigned char* dst, unsigned char* temp, int width, int height, int radio);

void ErosionTemplateSharedTwoSteps(unsigned char* src, unsigned char* dst, unsigned char* temp, int width, int height, int radio);
