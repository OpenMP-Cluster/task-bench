# ifndef STARPU_DIR
# $(error STARPU_DIR variable is not defined, aborting build)
# endif

DEBUG ?= 0

CC         = mpic++

CFLAGS     = -DADD_ -fPIC -std=c++11
LDFLAGS    = -Wall -std=c++11

STARPU_VERSION ?= 1.4

STARPU_DIR ?= /opt/starpu
#MPI_DIR    = /sw/openmpi/2.1.1

ifeq ($(strip $(DEBUG)),0)
CFLAGS		 += -O3
LDFLAGS		 += -O3
else
CFLAGS		 += -g -O0
LDFLAGS		 += -g -O0
endif

GPU_ARCH   ?= gfx90a
USE_STARPU_GPU ?= 1

ifeq ($(strip $(USE_STARPU_GPU)),1)
GPU_CFLAGS  = -fopenmp -foffload-lto --offload-arch=$(GPU_ARCH)
GPU_LDFLAGS = -fopenmp -foffload-lto --offload-arch=$(GPU_ARCH)
else
GPU_CFLAGS  = -fopenmp -foffload-lto -fopenmp-targets=x86_64-pc-linux-gnu
GPU_LDFLAGS = -fopenmp -foffload-lto -fopenmp-targets=x86_64-pc-linux-gnu
endif

CXX_GPU = clang++

# Include directories
INC        = -I$(STARPU_DIR)/include/starpu/$(STARPU_VERSION) -I../core
INC_EXT    = 

# Location of the libraries.
LIB        = -Wl,-rpath,$(STARPU_DIR)/lib: $(STARPU_DIR)/lib/libstarpu-$(STARPU_VERSION).so $(STARPU_DIR)/lib/libstarpumpi-$(STARPU_VERSION).so -L../core -lcore_s
LIB_EXT    =

ifeq ($(strip $(TASKBENCH_USE_HWLOC)),1)
  INC_EXT += -I$(HWLOC_DIR)/include
endif

INC := $(INC) $(INC_EXT)
LIB := $(LIB) $(LIB_EXT)

CFLAGS += $(INC)

include ../core/make_blas.mk

TARGET = main main_buffer_core main_expl
all: $(TARGET)

.PRECIOUS: %.c %.o

data.o: data.cc data.h
	$(CC) -c $(CFLAGS) $<

main.o: main.cc
	$(CC) -c $(CFLAGS) $<

main: main.o data.o
	$(CC) $^ $(LIB) $(LDFLAGS) -o $@ 

main_buffer_core.o: main_buffer_core.cc
	$(CC) -c $(CFLAGS) $<

main_buffer_core: main_buffer_core.o data.o
	$(CC) $^ $(LIB) $(LDFLAGS) -o $@ 

main_expl.o: main_expl.cc
	$(CC) -c $(CFLAGS) $<

main_expl: main_expl.o data.o
	$(CC) $^ $(LIB) $(LDFLAGS) -o $@

main_static.o: main_static.cc
	$(CC) -c $(CFLAGS) $<

main_static: main_static.o data.o
	$(CC) $^ $(LIB) $(LDFLAGS) -o $@

main_gpu.o: main_gpu.cc
	$(CXX_GPU) -c $(CFLAGS) -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -I/opt/rocm/include $(GPU_CFLAGS) $<

main_gpu: main_gpu.o data.o
	$(CXX_GPU) $^ $(LIB) $(GPU_LDFLAGS) -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -L/usr/local/mpich/lib/ -I/opt/rocm/include $(LDFLAGS) -lamdhip64 -lmpi -o $@

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean
