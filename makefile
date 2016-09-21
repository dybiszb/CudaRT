OBJ = main.o commons.o rayTrace.o
NVCC_FLAGS= -arch=sm_20 -Wall
L_FLAGS= -Xlinker -lGL -lGLU -lglut -lm

.PHONY=clean
.PHONY=all

all: main clean

main: ${OBJ}
	@echo "=== Linking executable: main ==="
	nvcc -o main ${OBJ} ${L_FLAGS}

main.o: main.cu
	@echo "=== Compiling main.cu file ==="
	nvcc -o main.o -c -arch=sm_20 -rdc=true main.cu

commons.o: commons.cu
	@echo "=== Compiling commons.cu file ==="
	nvcc -o commons.o -c -arch=sm_20 -rdc=true commons.cu

rayTrace.o: rayTrace.cu
	@echo "=== Compiling rayTrace.cu file ==="
	nvcc -o rayTrace.o -c -arch=sm_20 -rdc=true rayTrace.cu

clean:
	@echo "=== Cleaning ==="
	rm *.o
