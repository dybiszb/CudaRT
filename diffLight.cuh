#ifndef DIFFUSE_LIGHT_CUH
#define DIFFUSE_LIGHT_CUH

class DiffLight
{
    public:
        DiffLight(float x, float y, float z, float intensity) :
        x(x), y(y), z(z), intensity(intensity) {};
        float x, y, z;
        float intensity;
};

#endif