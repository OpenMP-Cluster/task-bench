DEBUG ?= 0
CUDA ?= 0

## -fopenmp-targets=x86_64-pc-linux-mpi mpirun -np 2
## gdb ./main lldb

ifeq ($(strip $(CUDA)),1)
CXXFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -v
LDFLAGS  = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -v
else
CXXFLAGS = -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu -v
LDFLAGS  = -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu -v
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

TARGET = main main_no_datamap main_multidevice
all: $(TARGET)

.PRECIOUS: %.cc %.o

main.o: main.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main: main.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@

main_no_datamap.o: main_no_datamap.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main_no_datamap: main_no_datamap.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@

main_multidevice.o: main_multidevice.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main_multidevice: main_multidevice.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean
