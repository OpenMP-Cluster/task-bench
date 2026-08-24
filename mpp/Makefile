DEBUG ?= 0
CC          ?= clang
CXX         ?= clang++
GPU_ARCH	?= gfx942
USE_MPP_GPU	 	?= 0

CXXFLAGS = -fopenmp -foffload-lto
LDFLAGS = -fopenmp -foffload-lto

## -fopenmp-targets=x86_64-pc-linux-mpi mpirun -np 2
## gdb ./main lldb

ifeq ($(strip $(USE_MPP_GPU)),1)
CXXFLAGS += --offload-arch=$(GPU_ARCH) 
LDFLAGS  += --offload-arch=$(GPU_ARCH)
else
CXXFLAGS += -fopenmp-targets=x86_64-pc-linux-gnu
LDFLAGS  += -fopenmp-targets=x86_64-pc-linux-gnu
endif

ifeq ($(strip $(DEBUG)),1)
CXXFLAGS += -g -O0
LDFLAGS  += -g -O0
else
CXXFLAGS += -O3
LDFLAGS  += -O3
endif

# Include directories
INC        = -I../core
INC_EXT    =

# Location of the libraries.
LIB        = -L../core -lcore_s
LIB_EXT    =

INC := $(INC) $(INC_EXT)
LIB := $(LIB) $(LIB_EXT)

CXXFLAGS += $(INC)

include ../core/make_blas.mk

TARGET = main main_buffer main_gpu
all: $(TARGET)

.PRECIOUS: %.cc %.o

main.o: main.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main: main.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@

main_buffer.o: main_buffer.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main_buffer: main_buffer.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@

main_gpu.o: main_gpu.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main_gpu: main_gpu.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean
