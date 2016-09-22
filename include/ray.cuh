#ifndef RAY_CUH
#define RAY_CUH

#include <iostream>
#include "vector3f.cuh"

using namespace std;

class Ray {
public:
    __host__ __device__
    Ray() {};
    __host__ __device__
    Ray(Vector3f E, Vector3f D) : E(E), D(D) { };
    Vector3f E;
    Vector3f D;

    __host__ __device__
    Vector3f P(float t) {
        //if (t <= 0) cout << "RAY: t <= 0\n";
        return E + (D * t);
    }
};

#endif
