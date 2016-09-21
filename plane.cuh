#ifndef PLANE_CUH
#define PLANE_CUH

#include "vector3f.cuh"

class Plane {
public:
    __host__ __device__ Plane(){};
    __host__ __device__ Plane(Vector3f N, Vector3f Q) : N(N), Q(Q) {};
    __host__ __device__ Plane(float level){
        Vector3f normal(0,1,0);
        Vector3f point(0,level,0);
        N = normal;
        Q = point;
    };
    Vector3f N;
    Vector3f Q;

    __host__ __device__ bool getT(Ray ray, float* t)
    {
        /* Ray parallel to the plane */
        if((N * ray.D) == 0) return false;

        //*t = (N.dot(N,(Q - ray.E))) / (N.dot(N,ray.D));
        *t = (N * (Q - ray.E)) / (N * ray.D);
        if(*t < 0) return false;
        else return true;
    }
};

#endif
