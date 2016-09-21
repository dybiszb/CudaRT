#include <host_defines.h>
#include "rayTrace.cuh"

__host__ __device__ bool getT(Ray r, Sphere s, float* t)
{
    Vector3f spherePos(s.x, s.y, s.z);
    Vector3f dist = spherePos - r.E;
    float B = r.D * dist;
    float D = B*B - dist * dist + s.radius * s.radius;
    if (D < 0.0f)
        return false;
    float t0 = B - sqrt(D);
    float t1 = B + sqrt(D);

    if ((t0 > 0.1f) && (t0 <= t1))
    {
        *t = (float)t0;

        return true;
    }
    if ((t1 > 0.1f) && (t1 < t0))
    {
        *t = (float)t1;
        return true;
    }
    return false;
}

__host__ __device__ RGB getPixel(unsigned char * texture, int x, int y) {
    if(x >=0 && y>=0 && x < 1200 && y < 764) {
        float R = texture[y * 1200 + x];
        float G = texture[(y * 1200 + x) + 1];
        float B = texture[(y * 1200 + x) + 2];
        return RGB(R,G,B);

    } else {
        return RGB(0,0,0);
    }
}

__host__ __device__ RGB trace(Ray ray, int depth, Sphere* spheres,Plane* plane, int n, DiffLight* light, Vector3f* eye, unsigned char * backgroundTex)
{
    float t = INF;
    float newT;
    int index = -1;
    bool isPlane = false;
    RGB background = RGB(0,0,0);

    /* Closest sphere with its intersection point */
    for(int i=0; i<n; i++) {
        if(getT(ray, spheres[i], &newT))
        {
            if(newT < t)
            {
                t = newT;
                index = i;
            }
        }
    }

    /* Plane with its intersection point */
    if(plane->getT(ray, &newT))
    {
        if(newT < t)
        {
            /* Check plane's borders */
            Vector3f point = ray.P(newT);
            if(point.x <= PLANE_X_TOP &&
               point.x >= PLANE_X_LOW &&
               point.z <= PLANE_Z_TOP &&
               point.z >= PLANE_Z_LOW)
            {
                t = newT;
                isPlane = true;
            }

        }
    }
    Vector3f pInter = ray.P(t);
    /* Get normal at this point */
    Vector3f spherePos(spheres[index].x, spheres[index].y, spheres[index].z);
    Vector3f N = pInter - spherePos;
    N.normalize();
    if(index != -1 || isPlane)
        return clip(shade(spheres[index],plane,isPlane , ray, pInter, N, depth, spheres, n, light, eye, backgroundTex));
    else
        return background;

}
__host__ __device__ Vector3f refract(Vector3f N, Vector3f incident, float n1, float  n2)
{

    float cosI = dot(incident, N);
	N.normalize();
    if(cosI > 0 ) {
       N = Vector3f(-N.x, -N.y, -N.z);
    } else {
       n1 = n2;
       n2 = 1.0f;
       cosI = - cosI;
    }
    float n = n1/n2;
    float sinT2 = 1.0 - n * n * (1.0 - cosI * cosI);
    float cosT = sqrt(1.0 - sinT2);
    Vector3f ret =  incident*n + N*(n * cosI - cosT);
    ret.normalize();
	return ret;
}

__host__ __device__ RGB shade(Sphere s,Plane* plane, bool isPlane, Ray ray, Vector3f pInter, Vector3f N,
          int depth,Sphere* spheres, int n, DiffLight* light, Vector3f* eye, unsigned char * backgroundTex)
{
    RGB color(0,0,0);
    Ray rRay, tRay, sRay;
    RGB rColor, tColor;
    float dummyT;
    float diffuse;
    float ambient = 0.2;
    float specular;
    float phongFactor= 0;
    bool inShadow = false;

    /* Calculate all needed vectors */
    Vector3f lightPos(light->x, light->y, light->z);
    Vector3f L = lightPos-pInter;
    L.normalize();


    Ray shadow;
    shadow.E = pInter;
    shadow.D = L;

    if(isPlane) N = plane->N;
    N.normalize();

    if(isPlane) pInter.y = plane->Q.y;

    Vector3f inv_incident = Vector3f(-ray.D.x, -ray.D.y, -ray.D.z);
    inv_incident.normalize();
    Vector3f reflected_ray = N * (2.0f * (N * inv_incident)) - inv_incident;
    reflected_ray.normalize();

    Vector3f V = *eye - pInter;
    V.normalize();


    /* Check if not in the shadow */
    for(int i = 0; i < n; i++)
    {
        inShadow = getT(shadow,spheres[i],&dummyT);
        if(inShadow) break;
    }
//

    if(!inShadow)
    {
        /* DIFFUSE */
        float diffDot = N * L;
        if(diffDot < 0) diffuse = 0;
        else diffuse = 0.7f * diffDot;

        /* SPECULAR */
        if(N * L < 0) {
            specular = 0;
        } else {
            Vector3f inv_L = Vector3f(-L.x, -L.y, -L.z);
            Vector3f reflected_L = N * (2.0f * (N * inv_L)) - inv_L;
            reflected_L.normalize();


            specular = (float) (0.7 * pow(reflected_L * V, 34));
        }

        if(!isPlane)
        phongFactor = (ambient + diffuse + specular);
        else phongFactor = (ambient + 0.4 * diffuse);
        phongFactor = clip(phongFactor, 0 ,1);


    } else {
        phongFactor = ambient;
    }





    if(isPlane){
        color.r += 100 * phongFactor;
        color.g += 100 * phongFactor;
        color.b += 100 * phongFactor;
    } else {
        color.r += s.R * phongFactor;
        color.g += s.G * phongFactor;
        color.b += s.B * phongFactor;
    };


    if(depth < TREE_DEPTH)
    {
        // REFLECTION
        if(isPlane) pInter.y = (float) (pInter.y + 0.02);
        rRay.E = pInter;

        if(!isPlane) rRay.D = reflected_ray;
        else rRay.D =reflected_ray;
        rColor = trace(rRay, depth + 1,spheres,plane, n, light, eye, backgroundTex);
        if(!isPlane){
            color.r += (0.5) * rColor.r;
            color.g += (0.5) * rColor.g;
            color.b += (0.5) * rColor.b;
        } else {
            color.r += (0.3) * rColor.r;
            color.g += (0.3) * rColor.g;
            color.b += (0.3) * rColor.b;
        }


        // REFRACTION

        ///////// TEMP REFRACTION //////////
        Vector3f refraction_N = N;
        float cosI = ray.D * refraction_N;
        float n1, n2;
        if (cosI > 0)
        {
            /* Incident and normal have the same direction, ray is inside the material. */
            n1 = 1.0;
            n2 = 1.1;

            /* Flip the normal around. */
            refraction_N = Vector3f(-refraction_N.x, -refraction_N.y, -refraction_N.z);
        }
        else
        {
            /* Incident and normal have opposite directions, so the ray is outside the material. */
            n2 = 1.0;
            n1 = 1.1;

            /* Make the cosine positive. */
            cosI = -cosI;
        }


        float cosT = (float) (1.0f - pow(n1 / n2, 2.0f) * (1.0f - pow(cosI, 2.0f)));


        /////////////////////////////////////////////////
        // TOTAL INTERNAL REFLECTION
        /////////////////////////////////////////////////
        if (cosT < 0.0f)
        {
            // TOTAL INTERNAL REFLECTION
            Vector3f biased_origin = pInter + refraction_N * 0.002;
            tRay.E = pInter;
            tRay.D = refraction_N * (2.0f * (refraction_N * inv_incident)) - inv_incident;
            tRay.D.normalize();
            tColor = trace(tRay, depth + 1, spheres,plane, n, light, eye, backgroundTex);
        }
        /////////////////////////////////////////////////
        // STANDARD REFRACTION
        /////////////////////////////////////////////////
        else {
            cosT = (float) sqrt(cosT);
            Vector3f biased_origin = pInter + refraction_N * 0.002;
            tRay.E = pInter;
            tRay.D = ray.D * (n1 / n2) + refraction_N * ((n1 / n2) * cosI - cosT);
            tRay.D.normalize();
            tColor = trace(tRay, depth + 1, spheres,plane, n, light, eye, backgroundTex);
        }

        ////////////////////////////////////

//        tRay.E = pInter;
//        tRay.D = refract(N, ray.D, 1, 1.83);
//        tRay.D.normalize();
//        tColor = trace(tRay, depth + 1, spheres,plane, n, light, eye);



        if(!isPlane){
            color.r += (0.5) * tColor.r;
            color.g += (0.5) * tColor.g;
            color.b += (0.5) * tColor.b;
        };
    }



    return color;
}


__global__ void kernel( Sphere *spheres, Plane* plane, DiffLight* light, unsigned char *pixels, int* n,
                        Vector3f* eye, Vector3f* pInter, unsigned char* background) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - SCREEN_WIDTH/2);
    float   oy = (y - SCREEN_HEIGHT/2);

    /* Calculate starting ray */
    Vector3f D(ox,oy,1001);
    D.normalize();
    Ray ray(*eye, D);

    RGB color = trace(ray, 1, spheres, plane, *n, light, eye, background);

    pixels[offset*4 + 0] = (int)(color.r);
    pixels[offset*4 + 1] = (int)(color.g);
    pixels[offset*4 + 2] = (int)(color.b);
    pixels[offset*4 + 3] = 255;
}
