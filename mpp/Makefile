DEBUG ?= 0
CUDA ?= 0

## -fopenmp-targets=x86_64-pc-linux-mpi mpirun -np 2
## gdb ./main lldb

ifeq ($(strip $(CUDA)),1)
CXXFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
LDFLAGS  = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
else
CXXFLAGS = -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu
LDFLAGS  = -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu
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
	$(CXX) -c $(CXXFLAGS) -foffload-lto -g $<

main_gpu: main_gpu.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -foffload-lto -g -o $@

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean
