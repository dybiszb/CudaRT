/**
 * author:  Bartlomiej Dybisz
 * subject: Graphic Processors in Computational Applications
 * title:   Recursive ray tracing animation for a set of spheres in real-time.
 */

#include <iostream>
#include <ctime>
#include <string>
#include <cstdlib>
#include <time.h>
#include "cuda.h"
#include <GL/glut.h>
#include "config.cuh"
#include "sphere.cuh"
#include "diffLight.cuh"
#include "vector3f.cuh"
#include "rgb.cuh"
#include "ray.cuh"
#include "plane.cuh"
#include "commons.cuh"
#include "rayTrace.cuh"

using namespace std;

/* ----- Global Variables ----- */
int                 gCalcType;
int                 gSpheresCount;
int                 gAnimation;
float               theta;
float               psi;
unsigned char*      gPixels;
Sphere*             gSpheres;
Plane*              gPlane;
DiffLight*          gLight;
ofstream            gLogStream;
Vector3f*           gView;
unsigned char *     gBackgroundTexture;
/* ----- ---------------- ----- */


/* ----- GLUT Related Functions ----- */
__host__            unsigned char * loadBackgroundTexture(const char * filename , int width, int height);
__host__            void initScene();
__host__            void render();
__host__            void keyPress(unsigned char key, int x, int y);
__host__            void mouse(int button, int state, int x, int y);
__host__            void initGlut(int *argc, char *argv[]);
__host__            void OnIdle(void);
__host__            void animationStep();
__host__            void setAcceleration(Sphere& sphere,Vector3f yGravity);
__host__            void eulerIntegrate(Sphere& sphere, float dt);
/* ----- ---------------------- ----- */

/* ----- CPU/GPU Calculations ----- */
__host__ void cpu_rt();
__host__ void gpu_rt();
/* ----- -------------------- ----- */

int main(int argc, char *argv[])
{
    srand(static_cast <unsigned> (time(0)));
    parseArgs(argc, argv, &gCalcType, &gSpheresCount);
    initScene();
    initGlut(&argc, argv);
    glutMainLoop();
    free(gSpheres);
    free(gBackgroundTexture);
    return EXIT_SUCCESS;
}

__host__  void render() {
    if(gAnimation == ON)        animationStep();
    if(gCalcType == CPU)        cpu_rt();
    else if (gCalcType == GPU)  gpu_rt();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, gPixels );
    glFlush();
}

__host__ void keyPress(unsigned char key, int x, int y)
{
    switch (key) {
        case KEY_W:
            gLight->y += STEP;
            render();
            break;
        case KEY_S:
            gLight->y -= STEP;
            render();
            break;
        case KEY_Q:
            gLight->x -= STEP;
            render();
            break;
        case KEY_E:
            gLight->x += STEP;
            render();
            break;
        case KEY_A:
            theta += 3.14/180;

            for(int i = 0; i < gSpheresCount; i++)
            {
                gSpheres[i].x = cos(theta) * gSpheres[i].o_x - sin(theta) * gSpheres[i].o_z;
                gSpheres[i].z = sin(theta) * gSpheres[i].o_x  + gSpheres[i].o_z *cos(theta);
            }

            render();
            break;
        case KEY_D:
            theta -= 3.14/180;

            for(int i = 0; i < gSpheresCount; i++)
            {
                gSpheres[i].x = cos(theta) * gSpheres[i].o_x - sin(theta) * gSpheres[i].o_z;
                gSpheres[i].z = sin(theta) * gSpheres[i].o_x  + gSpheres[i].o_z *cos(theta);
            }

            render();
            break;
        case KEY_SPACEBAR:
            cout << "[] Animation: " << ((gAnimation != ON) ? "ON" : "OFF") << "\n";
            gAnimation = (gAnimation == ON) ? OFF : ON;
            break;
        case ESCAPE:
            exit(0);
    }
}
__host__ void mouse(int button, int state, int x, int y)
{
    /* Wheel event */
    if ((button == 3) || (button == 4))
    {
        if (state == GLUT_UP) return; // Disregard redundant GLUT_UP events
        gView->z += (button == 3) ? 3*STEP : -3*STEP;
        render();
    }
    /* Normal event */
    else{
       // printf("Button %s At %d %d\n", (state == GLUT_DOWN) ? "Down" : "Up", x, y);
    }
}
__host__  void initGlut(int *argc, char *argv[])
{
    glutInit(argc, argv);
    controls();
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutCreateWindow(WINDOW_NAME);
    glutDisplayFunc(render);
    glutKeyboardFunc(keyPress);
    glutMouseFunc(mouse);
    glutIdleFunc(OnIdle);
}

__host__  unsigned char * loadBackgroundTexture(const char * filename , int width, int height) {
    unsigned char * data;

    FILE * file;

    file = fopen( filename, "rb" );
    if ( file == NULL ) return 0;
    data = (unsigned char *)malloc((size_t) (width * height * 3));

    fread( data, width * height * 3, 1, file );
    fclose( file );

    for(int i = 0; i < width * height ; ++i)
    {
        int index = i*3;
        unsigned char B,R;
        B = data[index];
        R = data[index+2];

        data[index] = R;
        data[index+2] = B;

    }

    return data;
}

__host__  void initScene()
{
    gPixels = new unsigned char[SCREEN_WIDTH * SCREEN_HEIGHT * 4];
    gSpheres = (Sphere*)malloc( sizeof(Sphere) * gSpheresCount );
    gPlane = new Plane(-300);
    gLight = new DiffLight(0,-250,-150,1.0);
    gView = new Vector3f(0,0,VIEW_START_Z);
    gAnimation = OFF;
    gBackgroundTexture = loadBackgroundTexture("sky.bmp", 1200, 764);

    for(int i = 0; i < gSpheresCount; ++i )
    {
        gSpheres[i].x       = randFloat(-SCREEN_WIDTH + 150,SCREEN_WIDTH - 150);
        gSpheres[i].y       = randFloat(-200, SCREEN_HEIGHT - 150);
        gSpheres[i].z       = randFloat(-700.0,700.0);
        // While rotating - original x,y,z are quite useful
        gSpheres[i].o_x     = gSpheres[i].x;
        gSpheres[i].o_y     = gSpheres[i].y;
        gSpheres[i].o_z     = gSpheres[i].z;

        gSpheres[i].R       = randFloat(255.0);
        gSpheres[i].G       = randFloat(255.0);
        gSpheres[i].B       = randFloat(255.0);
        gSpheres[i].radius  = randFloat(70.0f, 100.0f);
        gSpheres[i].volume = (4 * 3.14 * gSpheres[i].radius * gSpheres[i].radius * gSpheres[i].radius) /2;
        gSpheres[i].mass = gSpheres[i].volume * DENSITY;

        Vector3f vel(0,0,0);
        Vector3f acc(0,0,0);
        gSpheres[i].velocity = vel;
        gSpheres[i].acceleration = acc;
    }
}

__host__  void setAcceleration(Sphere& sphere,Vector3f yGravity)
{
    float x = yGravity.x - (DAMPNESS/sphere.mass * sphere.velocity.x);
    float y = yGravity.y - (DAMPNESS/sphere.mass * sphere.velocity.y);
    float z = yGravity.z - (DAMPNESS/sphere.mass * sphere.velocity.z);

    Vector3f newAcceleration(x,y,z);
    sphere.acceleration.x = newAcceleration.x;
    sphere.acceleration.y = newAcceleration.y;
    sphere.acceleration.z = newAcceleration.z;
}

__host__  void eulerIntegrate(Sphere& sphere, float dt)
{
    Vector3f pos(sphere.x, sphere.y, sphere.z);
    Vector3f vel;

    pos = pos + sphere.velocity * (dt);
    vel = sphere.velocity + sphere.acceleration * (dt);

    sphere.x = pos.x;
    sphere.y = pos.y;
    sphere.z = pos.z;
    sphere.velocity = vel;
    if(sphere.y < 0)
    printf("Sphere height:%f\n", sphere.y);
}

__host__  void animationStep()
{
    for(int i = 0; i < gSpheresCount; i++)
    {
        Vector3f yGravity(0,GRAVITY,0);
        Vector3f pStart(gSpheres[i].x, gSpheres[i].y, gSpheres[i].z);
        setAcceleration(gSpheres[i], yGravity);
        eulerIntegrate(gSpheres[i], _dt);
        if(gSpheres[i].y - gSpheres[i].radius < gPlane->Q.y) {
            Vector3f velocityTemp(gSpheres[i].velocity * -1);
            //float projection = velocityTemp.dot(velocityTemp,gPlane->N);
            float projection = velocityTemp * gPlane->N;
            Vector3f lengthVector = gPlane->N * projection;

            Vector3f reflection = lengthVector *(2.0);
            Vector3f velocityFinal = reflection + gSpheres[i].velocity;
            gSpheres[i].velocity = velocityFinal;
        }
    }
}

__host__  void OnIdle(void) {
    if(gAnimation == ON) {
        animationStep();
        glutPostRedisplay();
    }
}

__host__ void cpu_rt()
{
    clock_t begin = clock();

    for(int x = 0; x < SCREEN_WIDTH; x++)
    {
        for(int y = 0; y < SCREEN_HEIGHT; y++)
        {
            /* Calculate coordinates */
            float   ox = (x - SCREEN_WIDTH/2);
            float   oy = (y - SCREEN_HEIGHT/2);
            int offset = x + y * SCREEN_WIDTH;

            /* Prepare starting ray */
            Vector3f D(ox, oy, 1001);
            D.normalize();
            Ray ray(*gView, D);

            /* Shoot recursive ray */
            RGB color = trace(ray, 1, gSpheres,gPlane, gSpheresCount, gLight, gView, NULL);

            /* Update pixels */
            gPixels[offset*4 + 0] = (int)(color.r);
            gPixels[offset*4 + 1] = (int)(color.g);
            gPixels[offset*4 + 2] = (int)(color.b);
            gPixels[offset*4 + 3] = 255;
        }
    }

    clock_t end = clock();

    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    saveTime(elapsed_secs, &gCalcType, &gSpheresCount);
}

__host__ void gpu_rt()
{
    /* Initialize pointers */
    Sphere* spheres;
    DiffLight* light;
    unsigned char* pixels;
    unsigned char* backgroundTexture;
    int* n;
    Vector3f* eye;
    Vector3f* pInter;
    Plane* plane;

    /* Start time measuring */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    /* Initialize pointer on hardware */
    cudaMalloc((void**)&spheres, sizeof(Sphere) * gSpheresCount) ;
    cudaMalloc((void**)&light,   sizeof(DiffLight)) ;
    cudaMalloc((void**)&pixels,  SCREEN_WIDTH * SCREEN_HEIGHT * 4);
    cudaMalloc((void**)&backgroundTexture, 1200 * 764 * 3);
    cudaMalloc((void**)&n,       sizeof(int)) ;
    cudaMalloc((void**)&eye,     sizeof(Vector3f));
    cudaMalloc((void**)&pInter,  sizeof(Vector3f));
    cudaMalloc((void**)&plane,   sizeof(Plane));

    /* Upload information */
    cudaMemcpy(spheres, gSpheres,       sizeof(Sphere) * gSpheresCount, cudaMemcpyHostToDevice);
    cudaMemcpy(light,   gLight,         sizeof(DiffLight),              cudaMemcpyHostToDevice);
    cudaMemcpy(n,       &gSpheresCount, sizeof(int),                    cudaMemcpyHostToDevice);
    cudaMemcpy(eye,     gView,          sizeof(Vector3f),               cudaMemcpyHostToDevice);
    cudaMemcpy(plane,   gPlane,         sizeof(Plane),                  cudaMemcpyHostToDevice);
    cudaMemcpy(backgroundTexture, gBackgroundTexture, 1200 * 764 * 3,   cudaMemcpyHostToDevice);

    /* Run on hardware */
    dim3    grids(SCREEN_WIDTH/16,SCREEN_HEIGHT/16);
    dim3    threads(16,16);
    kernel<<<grids,threads>>>(spheres, plane, light, pixels, n, eye, pInter, backgroundTexture);

    /* Save updated info to cpu-image */
    cudaMemcpy( gPixels, pixels, 4 * SCREEN_WIDTH * SCREEN_HEIGHT, cudaMemcpyDeviceToHost);

    /* Stop time measuring */
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    saveTime(0.001 * elapsedTime, &gCalcType, &gSpheresCount);

    /* Dispose */
    cudaFree(pixels);
    cudaFree(spheres);
    cudaFree(n);
    cudaFree(light);
    cudaFree(eye);
    cudaFree(pInter);
    cudaFree(plane);
    cudaFree(backgroundTexture);
}
