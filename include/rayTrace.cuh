#ifndef RAY_TRACE_CUH
#define RAY_TRACE_CUH

#include "config.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "vector3f.cuh"
#include "diffLight.cuh"
#include "plane.cuh"
#include "rgb.cuh"
#include "commons.cuh"
__host__ __device__ RGB getPixel(unsigned char * texture, int x, int y);
__host__ __device__ bool getT(Ray r, Sphere s, float* t);
__host__ __device__ RGB trace(Ray ray, int depth, Sphere* spheres, Plane* plane,
                              int n, DiffLight* light, Vector3f* eye, unsigned char * backgroundTex);
__host__ __device__ RGB getTileColor(float x, float y) ;
__host__ __device__ RGB shade(Sphere s,Plane* plane, bool isPlane, Ray ray, Vector3f pInter,
                              Vector3f N, int depth, Sphere* spheres, int n,
                              DiffLight* light, Vector3f* eye,unsigned char * backgroundTex);
__host__ __device__ Vector3f refract(Vector3f N, Vector3f incident, float n1,float  n2);
__global__          void kernel(Sphere *spheres, Plane* plane, DiffLight* light, unsigned char *pixels, int* n,
                                Vector3f* eye, Vector3f* pInter, unsigned char* background);

#endif
