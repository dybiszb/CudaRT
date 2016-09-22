# Recursive Ray Tracing Animation for a Set of Spheres in Real-Time
## Project for 'Graphic Processors in Computational Applications' course at Warsaw University of Technology

## Technology:
- C
- CUDA

## Screenshots

_          |  Server Example
:-------------------------:|:-------------------------:
![alt tag](https://raw.githubusercontent.com/dybiszb/GPURayTracer/master/img/scr1.png)  |  ![alt tag](https://raw.githubusercontent.com/dybiszb/NetworkScrabble/master/img/server_scr.png)



## Compiling<a name="compile"></a>
Note: For obvious reasons only UNIX compilation is allowed.

In repository main folder call:

```
make all
```

Two executables should be created: server and client.

## Run<a name="run"></a>

First of all, run a server calling:
```
./server
```
in repository folder. At this point new clients can connect. Each of them
must be started in new console window via:
```
./client
```
If you want to break connection press CTRL-C. All controls are described
during the client application runtime.
