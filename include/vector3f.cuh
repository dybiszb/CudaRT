#ifndef VECTOR_3F_CUH
#define VECTOR_3F_CUH

#include <math.h>

class Vector3f {
public:
    __host__ __device__
    Vector3f() : x(0), y(0), z(0) {};
    __host__ __device__
    Vector3f(float x, float y, float z):x(x), y(y), z(z) {};

    float x, y, z;

    __host__ __device__
    void normalize()
    {
        float length = sqrt(x*x + y*y+z*z);
        x = x / length;
        y = y / length;
        z = z / length;
    }

    __host__ __device__
    Vector3f operator - (const Vector3f& rhs)
    {
        return subtract(*this, rhs);
    }

    __host__ __device__
    Vector3f operator + (const Vector3f& rhs)
    {
        return add(*this, rhs);
    }

    __host__ __device__
    Vector3f operator * (float scalar)
    {
        return Vector3f(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__
    float dot(const Vector3f& lhs, const Vector3f& rhs)
    {
        return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
    }

    __host__ __device__
    Vector3f cross(const Vector3f& lhs, const Vector3f& rhs)
    {
        Vector3f crossProduct;

        crossProduct.x = (lhs.y * rhs.z) - (lhs.z * rhs.y);
        crossProduct.y = (lhs.z * rhs.x) - (lhs.x * rhs.z);
        crossProduct.z = (lhs.x * rhs.y) - (lhs.y * rhs.x);

        return crossProduct;
    }

    __host__ __device__
    Vector3f add(const Vector3f& lhs, const Vector3f& rhs)
    {
        return Vector3f(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
    }

    __host__ __device__
    Vector3f subtract(const Vector3f& lhs, const Vector3f& rhs)
    {
        return Vector3f(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }

    __host__ __device__
    float operator * (const Vector3f& rhs)
    {
        return dot(*this,rhs);
    }

};

#endif