# Recursive Ray Tracing Animation for a Set of Spheres in Real-Time
## Project for 'Graphic Processors in Computational Applications' course at Warsaw University of Technology

## Technology:
- C
- CUDA

## Screenshots

Img1        | Img2  
:-------------------------:|:-------------------------:
![alt tag](https://raw.githubusercontent.com/dybiszb/GPURayTracer/master/img/scr1.png)  |  ![alt tag](https://raw.githubusercontent.com/dybiszb/GPURayTracer/master/img/scr2.png)



## Compiling<a name="compile"></a>
Note: For obvious reasons only UNIX compilation is allowed.

In repository main folder call:

```
make all
```

Two executables should be created: server and client.

## Run<a name="run"></a>
 
To start with the application can work in two different modes:
- CPU - runs calculations on a 'classic' processor.
- GPU - runs calculations on a 'graphics' processor.

In order to run the former call:
```
./main cpu [number of spheres]
```

The latter one can be invoked via:
```
./main gpu [number of spheres]
```

