#include "../include/commons.cuh"

__host__  void usage()
{
    cout << "./main [cpu/gpu] [number_of_spheres]\n";
    exit(EXIT_FAILURE);
}

__host__  void controls()
{
    cout << "[] Controls:\n";
    cout << "|- View  Position:         A, D, Mouse-Scroll\n";
    cout << "|- Light Position:         W, S, Q, E\n";
    cout << "|- Start/Stop Animation:   SPACEBAR\n";
    cout << "|- Exit:                   ESC\n";
}

__host__  void parseArgs(int argc, char* argv[],int* gCalcType, int* gSpheresCount)
{
    if(argc != 3) usage();
    else
    {
        if(!strcmp(argv[1],"cpu")) *gCalcType = CPU;
        else if(!strcmp(argv[1],"gpu")) *gCalcType = GPU;
        else usage();
        *gSpheresCount = atoi(argv[2]);
    }
}

__host__  void saveTime(double time, int* gCalcType, int* gSpheresCount)
{
    ofstream            gLogStream;
    string calcType = (*gCalcType == GPU) ? "gpu" : "cpu";
    gLogStream.open(LOG_FILE, std::ios_base::app);
    gLogStream << "calc_type: " << calcType << " spheres: "<< *gSpheresCount << " Time: " << time << endl;
}

__host__  float randFloat(float rBoundary)
{
    return static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/rBoundary));
}

__host__  float randFloat(float lBoundary, float rBoundary)
{
    return lBoundary + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(rBoundary-lBoundary)));
}

__host__ __device__ float dot(Vector3f lhs, Vector3f rhs)
{
    return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

__host__ __device__ float clip(float val, float down, float up)
{
    val = (val > up) ? up : val;
    val = (val < down) ? down : val;
    return val;
}

__host__ __device__ RGB clip(RGB color)
{
    color.r = clip(color.r, 0 ,255);
    color.g = clip(color.g, 0 ,255);
    color.b = clip(color.b, 0 ,255);
    return color;
}
