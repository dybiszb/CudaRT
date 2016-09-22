#ifndef RGB_CUH
#define RGB_CUH

class RGB {
public:
    __host__ __device__
    RGB() {};
    __host__ __device__
    RGB(float r, float g, float b) : r(r), g(g), b(b) {};
    float r;
    float g;
    float b;
};

#endif