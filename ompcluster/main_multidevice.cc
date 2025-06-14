// C headers
#include <assert.h>
#include <omp.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

// C++ headers
#include <algorithm>
#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
#include <memory>

// Project headers
#include "core.h"
#include "timer.h"

#define MAX_NUM_ARGS 10

/*
Structure to store the position of a task in the task graph.
  x: represents the column
  y: represents the row
*/
struct task_args_t {
  int x;
  int y;
};

/*
Structure to store the information of a task
  output_buff: Stores the output information of the task when it finishes being executed.
*/
struct tile_t {
  float dep;
  char *output_buff;
};

/*
Structure to store the task graph 
  data: Array that contains the pointers for the information of each task
  M: Heigth
  N: Width
*/
struct matrix_t {
  tile_t *data;
  int M;
  int N;
};

/*
Structure to manager the pool of target devices available in the env
This structure provide a function to get the next device available
for use in a omp target task.
*/
struct device_pool_t {
  int num_devices;
  int current_device;

  device_pool_t():
    num_devices(omp_get_num_devices()), current_device(-1) {}

  int get_next_device_rr() {
    // Get a device using round-robin strategy
    current_device = (current_device + 1) % num_devices;
    return current_device;
  }

  int get_device(int x, int y) {
    // Currently, each node expose 4 CPU device to MPI Plugin
    // but it should be only 1
    // TODO: Fix device count problem on MPI Proxy Plugin
    // TODO: Improve this function to schedule the task
    // to the best possible device using 2D coordinate
    return (x % (num_devices));
  }
};


/*
Function to call the kernel to execute the task.
@param[in] argv_new
  Char array containing the arguments, one for each string passed on the command line.
@param[in] argc
  An integer value that indicates the number of arguments in the command line.
@param[in] graph_idx
  The index of the task graph that the task to be executed belongs to.
@param[in] x0
  Represents the column of the executed task in the task graph.
@param[in] y0
  Represents the row of the executed task in the task graph.
@param[in] num_args
  Number of task dependencies in position (y0,x0).
@param[in] input_ptrs
  The input data for the task (y0,x0).
@param[in,out] output_ptr
  On entry, the data stored in task (y0,x0).
  On exit, the new data of the task (y0,x0), after executing
*/

void task_execute(const TaskGraph &graph, int x0, int y0, int num_args,
                  char **input_ptrs, char *output_ptr)
{
  // Allocate extra memory to run the task
  size_t scratch_bytes = graph.scratch_bytes_per_task;
  auto scratch_buf = std::make_unique<char[]>(scratch_bytes);
  char *scratch_ptr = scratch_buf.get();
  assert(scratch_ptr);
  TaskGraph::prepare_scratch(scratch_ptr, scratch_bytes);
  size_t output_bytes = graph.output_bytes_per_task;

  // Designate the size of the input data to run the task.
  size_t input_bytes[num_args];
  for (int idx_bytes = 0; idx_bytes < num_args; ++idx_bytes) {
    input_bytes[idx_bytes] = graph.output_bytes_per_task;
  }

  // Call the core.h function to execute task
  graph.execute_point(y0, x0, output_ptr, output_bytes,
                      (const char **)input_ptrs, input_bytes, num_args,
                      scratch_ptr, graph.scratch_bytes_per_task);
}

int main(int argc, char ** argv)
{
  // Create the app to store the information of the Task Graphs
  App new_app(argc, argv);  
  std::vector<TaskGraph> graphs = new_app.graphs;

  device_pool_t device_pool;

  // Create the Task Graphs and the information of each task
  matrix_t *matrix = (matrix_t *)malloc(sizeof(matrix_t) * graphs.size());
  for(unsigned i = 0; i<graphs.size(); i++){
    TaskGraph &graph = graphs[i];    
    matrix[i].M = graph.nb_fields;
    matrix[i].N = graph.max_width;
    int number_point= matrix[i].M * matrix[i].N;
    matrix[i].data = (tile_t*)malloc(sizeof(tile_t) * number_point);  
    for (int j = 0; j < number_point; j++) {
      matrix[i].data[j].output_buff = (char *)malloc(sizeof(char) * graph.output_bytes_per_task);
    }
  }

  // Show the information of the task graphs
  new_app.display();

  double start = omp_get_wtime(); 

  // Run the task graph in parallel
  #pragma omp parallel
  #pragma omp master
  {
    for (unsigned graph_idx = 0; graph_idx < graphs.size(); graph_idx++) {
      const TaskGraph &graph = graphs[graph_idx];
      tile_t *mat = matrix[graph_idx].data;

      for (int y = 0; y < graph.timesteps; y++) {
        long offset = graph.offset_at_timestep(y);
        long width = graph.width_at_timestep(y);
        long dset = graph.dependence_set_at_timestep(y);
        int nb_fields = graph.nb_fields;

        task_args_t args[MAX_NUM_ARGS];
        int num_args = 0;
        int curr_arg_idx = 0;

        for (int x = offset; x <= offset+width-1; x++) {
          std::vector<std::pair<long, long> > deps = graph.dependencies(dset, x);
          num_args = 0;
          curr_arg_idx = 0;

          // Calculate the number of dependencies of the task in position (y,x)
          num_args = 1;
          args[curr_arg_idx].x = x;
          args[curr_arg_idx].y = y % nb_fields;
          curr_arg_idx++;
          if (deps.size() != 0 && y != 0) {
            long last_offset = graph.offset_at_timestep(y - 1);
            long last_width = graph.width_at_timestep(y - 1);
            for (std::pair<long, long> dep : deps) {
              num_args += dep.second - dep.first + 1;
              for (int i = dep.first; i <= dep.second; i++) {
                if (i >= last_offset && i < last_offset + last_width) {
                  args[curr_arg_idx].x = i;
                  args[curr_arg_idx].y = (y - 1) % nb_fields;
                  curr_arg_idx++;
                } else {
                  num_args--;
                }
              }
            }
          }
          assert(num_args == curr_arg_idx);

          // Calculate the position of the task to execute
          int x0 = args[0].x;
          int y0 = args[0].y;
          int point0 = y0 * graph.max_width + x0;

          // Get the pointer to the task output data
          tile_t *tile_in0 = &mat[point0];
          char *output_ptr = (char*)tile_in0->output_buff;

          // Get the id of the next device available to execute the task
          int device_id = device_pool.get_device(x0, y0);

          // Depending on the number of dependencies
          switch(num_args)
          {
            case 1:
            {
              #pragma omp target nowait \
                depend(inout: output_ptr)\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {
                char * input_ptrs[num_args];
                input_ptrs[0] = output_ptr;
                task_execute(graph,x0,y0, num_args, input_ptrs, output_ptr);
              }
              break;
            }
            case 2:
            {
              //Get the pointers to the tak input data
              int point1 = args[1].y * graph.max_width+ args[1].x;
              tile_t *tile_in1 = &mat[point1];
              char *input_ptr0 = (char*)tile_in1->output_buff;

              // Pragma to send the task a node to execute it
              #pragma omp target nowait \
                depend(inout: output_ptr)\
                depend(in: input_ptr0)\
                map(to:input_ptr0[:graph.output_bytes_per_task])\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {            
                // Join the input data into a single variable
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                // Execute the task
                task_execute(graph,x0,y0, num_args-1, input_ptrs, output_ptr);
              }
              break;
            }
            case 3:
            {
              int point1 = args[1].y * graph.max_width+ args[1].x;
              int point2 = args[2].y * graph.max_width+ args[2].x;

              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];

              char *input_ptr0 = (char*)tile_in1->output_buff;
              char *input_ptr1 = (char*)tile_in2->output_buff;

              #pragma omp target nowait \
                depend(inout: output_ptr)\
                depend(in: input_ptr0)\
                depend(in: input_ptr1)\
                map(to: input_ptr0[:graph.output_bytes_per_task],\
                       input_ptr1[:graph.output_bytes_per_task])\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {                 
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                task_execute(graph,x0,y0, num_args-1, input_ptrs, output_ptr);
              }
              break;
            }
            case 4:
            {
              int point1 = args[1].y * graph.max_width+ args[1].x;
              int point2 = args[2].y * graph.max_width+ args[2].x;
              int point3 = args[3].y * graph.max_width+ args[3].x;

              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];
              tile_t *tile_in3 = &mat[point3];

              char *input_ptr0 = (char*)tile_in1->output_buff;
              char *input_ptr1 = (char*)tile_in2->output_buff;
              char *input_ptr2 = (char*)tile_in3->output_buff;

              #pragma omp target nowait \
                depend(inout: output_ptr)\
                depend(in: input_ptr0)\
                depend(in: input_ptr1)\
                depend(in: input_ptr2)\
                map(to: input_ptr0[:graph.output_bytes_per_task],\
                       input_ptr1[:graph.output_bytes_per_task],\
                       input_ptr2[:graph.output_bytes_per_task])\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {                
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                task_execute(graph,x0,y0, num_args-1, input_ptrs, output_ptr);
              }
              break;
            }
            case 5:
            {
              int point1 = args[1].y * graph.max_width+ args[1].x;
              int point2 = args[2].y * graph.max_width+ args[2].x;
              int point3 = args[3].y * graph.max_width+ args[3].x;
              int point4 = args[4].y * graph.max_width+ args[4].x;

              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];
              tile_t *tile_in3 = &mat[point3];
              tile_t *tile_in4 = &mat[point4];

              char *input_ptr0 = (char*)tile_in1->output_buff;
              char *input_ptr1 = (char*)tile_in2->output_buff;
              char *input_ptr2 = (char*)tile_in3->output_buff;
              char *input_ptr3 = (char*)tile_in4->output_buff;

              #pragma omp target nowait \
                depend(inout: output_ptr)\
                depend(in: input_ptr0)\
                depend(in: input_ptr1)\
                depend(in: input_ptr2)\
                depend(in: input_ptr3)\
                map(to: input_ptr0[:graph.output_bytes_per_task],\
                        input_ptr1[:graph.output_bytes_per_task],\
                        input_ptr2[:graph.output_bytes_per_task],\
                        input_ptr3[:graph.output_bytes_per_task])\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                input_ptrs[3] = input_ptr3;
                task_execute(graph,x0,y0, num_args-1, input_ptrs, output_ptr);
              }
              break;
            }
            case 6:
            {
              int point1 = args[1].y * graph.max_width+ args[1].x;
              int point2 = args[2].y * graph.max_width+ args[2].x;
              int point3 = args[3].y * graph.max_width+ args[3].x;
              int point4 = args[4].y * graph.max_width+ args[4].x;
              int point5 = args[5].y * graph.max_width+ args[5].x;

              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];
              tile_t *tile_in3 = &mat[point3];
              tile_t *tile_in4 = &mat[point4];
              tile_t *tile_in5 = &mat[point5];

              char *input_ptr0 = (char*)tile_in1->output_buff;
              char *input_ptr1 = (char*)tile_in2->output_buff;
              char *input_ptr2 = (char*)tile_in3->output_buff;
              char *input_ptr3 = (char*)tile_in4->output_buff;
              char *input_ptr4 = (char*)tile_in5->output_buff;

              #pragma omp target nowait \
                depend(inout: output_ptr)\
                depend(in: input_ptr0)\
                depend(in: input_ptr1)\
                depend(in: input_ptr2)\
                depend(in: input_ptr3)\
                depend(in: input_ptr4)\
                map(to: input_ptr0[:graph.output_bytes_per_task],\
                        input_ptr1[:graph.output_bytes_per_task],\
                        input_ptr2[:graph.output_bytes_per_task],\
                        input_ptr3[:graph.output_bytes_per_task],\
                        input_ptr4[:graph.output_bytes_per_task])\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {           
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                input_ptrs[3] = input_ptr3;
                input_ptrs[4] = input_ptr4;
                task_execute(graph,x0,y0, num_args-1, input_ptrs, output_ptr);
              }
              break;
            }
            case 7:
            {
              int point1 = args[1].y * graph.max_width+ args[1].x;
              int point2 = args[2].y * graph.max_width+ args[2].x;
              int point3 = args[3].y * graph.max_width+ args[3].x;
              int point4 = args[4].y * graph.max_width+ args[4].x;
              int point5 = args[5].y * graph.max_width+ args[5].x;
              int point6 = args[6].y * graph.max_width+ args[6].x;

              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];
              tile_t *tile_in3 = &mat[point3];
              tile_t *tile_in4 = &mat[point4];
              tile_t *tile_in5 = &mat[point5];
              tile_t *tile_in6 = &mat[point6];

              char *input_ptr0 = (char*)tile_in1->output_buff;
              char *input_ptr1 = (char*)tile_in2->output_buff;
              char *input_ptr2 = (char*)tile_in3->output_buff;
              char *input_ptr3 = (char*)tile_in4->output_buff;
              char *input_ptr4 = (char*)tile_in5->output_buff;
              char *input_ptr5 = (char*)tile_in6->output_buff;

              #pragma omp target nowait \
                depend(inout: output_ptr)\
                depend(in: input_ptr0)\
                depend(in: input_ptr1)\
                depend(in: input_ptr2)\
                depend(in: input_ptr3)\
                depend(in: input_ptr4)\
                depend(in: input_ptr5)\
                map(to: input_ptr0[:graph.output_bytes_per_task],\
                        input_ptr1[:graph.output_bytes_per_task],\
                        input_ptr2[:graph.output_bytes_per_task],\
                        input_ptr3[:graph.output_bytes_per_task],\
                        input_ptr4[:graph.output_bytes_per_task],\
                        input_ptr5[:graph.output_bytes_per_task])\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {           
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                input_ptrs[3] = input_ptr3;
                input_ptrs[4] = input_ptr4;
                input_ptrs[5] = input_ptr5;
                task_execute(graph,x0,y0, num_args-1, input_ptrs, output_ptr);
              }
              break;
            }
            case 8:
            {
              int point1 = args[1].y * graph.max_width+ args[1].x;
              int point2 = args[2].y * graph.max_width+ args[2].x;
              int point3 = args[3].y * graph.max_width+ args[3].x;
              int point4 = args[4].y * graph.max_width+ args[4].x;
              int point5 = args[5].y * graph.max_width+ args[5].x;
              int point6 = args[6].y * graph.max_width+ args[6].x;
              int point7 = args[7].y * graph.max_width+ args[7].x;
              
              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];
              tile_t *tile_in3 = &mat[point3];
              tile_t *tile_in4 = &mat[point4];
              tile_t *tile_in5 = &mat[point5];
              tile_t *tile_in6 = &mat[point6];
              tile_t *tile_in7 = &mat[point7];

              char *input_ptr0 = (char*)tile_in1->output_buff;
              char *input_ptr1 = (char*)tile_in2->output_buff;
              char *input_ptr2 = (char*)tile_in3->output_buff;
              char *input_ptr3 = (char*)tile_in4->output_buff;
              char *input_ptr4 = (char*)tile_in5->output_buff;
              char *input_ptr5 = (char*)tile_in6->output_buff;
              char *input_ptr6 = (char*)tile_in7->output_buff;

              #pragma omp target nowait \
                depend(inout: output_ptr)\
                depend(in: input_ptr0)\
                depend(in: input_ptr1)\
                depend(in: input_ptr2)\
                depend(in: input_ptr3)\
                depend(in: input_ptr4)\
                depend(in: input_ptr5)\
                depend(in: input_ptr6)\
                map(to: input_ptr0[:graph.output_bytes_per_task],\
                        input_ptr1[:graph.output_bytes_per_task],\
                        input_ptr2[:graph.output_bytes_per_task],\
                        input_ptr3[:graph.output_bytes_per_task],\
                        input_ptr4[:graph.output_bytes_per_task],\
                        input_ptr5[:graph.output_bytes_per_task],\
                        input_ptr6[:graph.output_bytes_per_task])\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {     
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                input_ptrs[3] = input_ptr3;
                input_ptrs[4] = input_ptr4;
                input_ptrs[5] = input_ptr5;
                input_ptrs[6] = input_ptr6;
                task_execute(graph,x0,y0, num_args-1, input_ptrs, output_ptr);
              }
              break;
            }
            case 9:
            {
              int point1 = args[1].y * graph.max_width+ args[1].x;
              int point2 = args[2].y * graph.max_width+ args[2].x;
              int point3 = args[3].y * graph.max_width+ args[3].x;
              int point4 = args[4].y * graph.max_width+ args[4].x;
              int point5 = args[5].y * graph.max_width+ args[5].x;
              int point6 = args[6].y * graph.max_width+ args[6].x;
              int point7 = args[7].y * graph.max_width+ args[7].x;
              int point8 = args[8].y * graph.max_width+ args[8].x;

              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];
              tile_t *tile_in3 = &mat[point3];
              tile_t *tile_in4 = &mat[point4];
              tile_t *tile_in5 = &mat[point5];
              tile_t *tile_in6 = &mat[point6];
              tile_t *tile_in7 = &mat[point7];
              tile_t *tile_in8 = &mat[point8];

              char *input_ptr0 = (char*)tile_in1->output_buff;
              char *input_ptr1 = (char*)tile_in2->output_buff;
              char *input_ptr2 = (char*)tile_in3->output_buff;
              char *input_ptr3 = (char*)tile_in4->output_buff;
              char *input_ptr4 = (char*)tile_in5->output_buff;
              char *input_ptr5 = (char*)tile_in6->output_buff;
              char *input_ptr6 = (char*)tile_in7->output_buff;
              char *input_ptr7 = (char*)tile_in8->output_buff;

              #pragma omp target nowait \
                depend(inout: output_ptr)\
                depend(in: input_ptr0)\
                depend(in: input_ptr1)\
                depend(in: input_ptr2)\
                depend(in: input_ptr3)\
                depend(in: input_ptr4)\
                depend(in: input_ptr5)\
                depend(in: input_ptr6)\
                depend(in: input_ptr7)\
                map(to: input_ptr0[:graph.output_bytes_per_task],\
                        input_ptr1[:graph.output_bytes_per_task],\
                        input_ptr2[:graph.output_bytes_per_task],\
                        input_ptr3[:graph.output_bytes_per_task],\
                        input_ptr4[:graph.output_bytes_per_task],\
                        input_ptr5[:graph.output_bytes_per_task],\
                        input_ptr6[:graph.output_bytes_per_task],\
                        input_ptr7[:graph.output_bytes_per_task])\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {       
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                input_ptrs[3] = input_ptr3;
                input_ptrs[4] = input_ptr4;
                input_ptrs[5] = input_ptr5;
                input_ptrs[6] = input_ptr6;
                input_ptrs[7] = input_ptr7;
                task_execute(graph,x0,y0, num_args-1, input_ptrs, output_ptr);
              }
              break;
            }
            case 10:
            {
              int point1 = args[1].y * graph.max_width+ args[1].x;
              int point2 = args[2].y * graph.max_width+ args[2].x;
              int point3 = args[3].y * graph.max_width+ args[3].x;
              int point4 = args[4].y * graph.max_width+ args[4].x;
              int point5 = args[5].y * graph.max_width+ args[5].x;
              int point6 = args[6].y * graph.max_width+ args[6].x;
              int point7 = args[7].y * graph.max_width+ args[7].x;
              int point8 = args[8].y * graph.max_width+ args[8].x;
              int point9 = args[9].y * graph.max_width+ args[9].x;

              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];
              tile_t *tile_in3 = &mat[point3];
              tile_t *tile_in4 = &mat[point4];
              tile_t *tile_in5 = &mat[point5];
              tile_t *tile_in6 = &mat[point6];
              tile_t *tile_in7 = &mat[point7];
              tile_t *tile_in8 = &mat[point8];
              tile_t *tile_in9 = &mat[point9];

              char *input_ptr0 = (char*)tile_in1->output_buff;
              char *input_ptr1 = (char*)tile_in2->output_buff;
              char *input_ptr2 = (char*)tile_in3->output_buff;
              char *input_ptr3 = (char*)tile_in4->output_buff;
              char *input_ptr4 = (char*)tile_in5->output_buff;
              char *input_ptr5 = (char*)tile_in6->output_buff;
              char *input_ptr6 = (char*)tile_in7->output_buff;
              char *input_ptr7 = (char*)tile_in8->output_buff;
              char *input_ptr8 = (char*)tile_in9->output_buff;

              #pragma omp target nowait \
                depend(inout: output_ptr)\
                depend(in: input_ptr0)\
                depend(in: input_ptr1)\
                depend(in: input_ptr2)\
                depend(in: input_ptr3)\
                depend(in: input_ptr4)\
                depend(in: input_ptr5)\
                depend(in: input_ptr6)\
                depend(in: input_ptr7)\
                depend(in: input_ptr8)\
                map(to: input_ptr0[:graph.output_bytes_per_task],\
                        input_ptr1[:graph.output_bytes_per_task],\
                        input_ptr2[:graph.output_bytes_per_task],\
                        input_ptr3[:graph.output_bytes_per_task],\
                        input_ptr4[:graph.output_bytes_per_task],\
                        input_ptr5[:graph.output_bytes_per_task],\
                        input_ptr6[:graph.output_bytes_per_task],\
                        input_ptr7[:graph.output_bytes_per_task],\
                        input_ptr8[:graph.output_bytes_per_task])\
                map(tofrom: output_ptr[:graph.output_bytes_per_task])\
                device(device_id)
              {        
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                input_ptrs[3] = input_ptr3;
                input_ptrs[4] = input_ptr4;
                input_ptrs[5] = input_ptr5;
                input_ptrs[6] = input_ptr6;
                input_ptrs[7] = input_ptr7;
                input_ptrs[8] = input_ptr8;
                task_execute(graph,x0,y0, num_args-1, input_ptrs, output_ptr);
              }
              break;
            }
            default:
              assert(false && "unexpected num_args");
          }
        }
      }
    }
  }

  double end = omp_get_wtime();
  double elapsed = end -start;
  // Show the report of how the implementation was executed
  new_app.report_timing(elapsed);

  // Deallocate memory block used to storethe information of the Task Graph
  for (unsigned i = 0; i < graphs.size(); i++) {
    for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
      free(matrix[i].data[j].output_buff);
      matrix[i].data[j].output_buff = NULL;
    }
    free(matrix[i].data);
    matrix[i].data = NULL;
  }
  
  free(matrix);
  matrix = NULL;
  return 0;
}