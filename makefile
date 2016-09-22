OBJ = main.o commons.o rayTrace.o
SOURCES = src
NVCC_FLAGS= -arch=sm_20 -Wall
L_FLAGS= -Xlinker -lGL -lGLU -lglut -lm

.PHONY=clean
.PHONY=all

all: main clean

main: ${OBJ}
	@echo "=== Linking executable: main ==="
	nvcc -o main ${OBJ} ${L_FLAGS}

main.o: ${SOURCES}/main.cu
	@echo "=== Compiling main.cu file ==="
	nvcc -o main.o -c -arch=sm_20 -rdc=true ${SOURCES}/main.cu

commons.o: ${SOURCES}/commons.cu
	@echo "=== Compiling commons.cu file ==="
	nvcc -o commons.o -c -arch=sm_20 -rdc=true ${SOURCES}/commons.cu

rayTrace.o: ${SOURCES}/rayTrace.cu
	@echo "=== Compiling rayTrace.cu file ==="
	nvcc -o rayTrace.o -c -arch=sm_20 -rdc=true ${SOURCES}/rayTrace.cu

clean:
	@echo "=== Cleaning ==="
	rm ${OBJ}
