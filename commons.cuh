#ifndef COMMONS_CUH
#define COMMONS_CUH

#include <string>
#include <iostream>
#include <fstream>
#include "config.cuh"
#include "vector3f.cuh"
#include "rgb.cuh"

using namespace std;

__host__            void    usage();
__host__            void    controls();
__host__            void    parseArgs(int argc, char* argv[], int* gCalcType, int* gSpheresCount);
__host__            void    saveTime(double time, int* gCalcType, int* gSpheresCount);
__host__            float   randFloat(float rBoundary);
__host__            float   randFloat(float lBoundary, float rBoundary);
__host__ __device__ float   dot(Vector3f vec1, Vector3f vec2);
__host__ __device__ float   clip(float val, float down, float up);
__host__ __device__ RGB     clip(RGB color);

#endif
