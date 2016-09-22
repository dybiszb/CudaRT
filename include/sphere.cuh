#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "diffLight.cuh"
#include "vector3f.cuh"

#define INF					2e10f
#define DENSITY             8.02
class Sphere {
public:
    Sphere() {
        Vector3f vel(0,0,0);
        Vector3f acc(0,0,0);
        velocity = vel;
        acceleration = acc;
    }
    Sphere(float x, float y, float z, float radius) :
            x(x), y(y), z(z), radius(radius) {
        volume = (4 * 3.14 * radius * radius * radius) /2;
        mass = volume * DENSITY;
        Vector3f vel(0,0,0);
        Vector3f acc(0,0,0);
        velocity = vel;
        acceleration = acc;
    }
    float radius;
    float x, y, z;
    float o_x, o_y, o_z;
    float R, G, B;
    float mass;
    float volume;
    Vector3f acceleration;
    Vector3f velocity;
};

#endif