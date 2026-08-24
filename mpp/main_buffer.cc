// C headers
#include <assert.h>
#include <omp.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

// C++ headers
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include <vector>

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
  omp_depend_t dep;
  int device_id;
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
  App app(argc, argv);  
  std::vector<TaskGraph> graphs = app.graphs;

  int num_devices = omp_get_num_devices();

  // Create the Task Graphs and the information of each task
  matrix_t *matrix = (matrix_t *)malloc(sizeof(matrix_t) * graphs.size());

  std::vector<std::vector<char *>> unique_outputs(graphs.size());

  for(unsigned i = 0; i<graphs.size(); i++){
    TaskGraph &graph = graphs[i];
    matrix[i].M = graph.nb_fields;
    matrix[i].N = graph.max_width;
    int number_point = matrix[i].M * matrix[i].N;
    matrix[i].data = (tile_t*)malloc(sizeof(tile_t) * number_point);

    // Minimum number of unique buffers needed to execute a task (conservative choice)
    int num_outputs = 2 * graph.max_width;
    int device_id, output_id;
    // Store unique buffers ptrs for this Task Graph
    unique_outputs[i].resize(num_outputs);

    for (int j = 0; j < num_outputs; j++)
      unique_outputs[i][j] = nullptr;

    // #pragma omp parallel for num_threads(num_devices)
    for (int j = 0; j < number_point; j++) {
      device_id = j % num_devices;
      output_id = j % num_outputs;
      if (unique_outputs[i][output_id] == nullptr) {
        unique_outputs[i][output_id] = (char *) omp_target_alloc(sizeof(char) * graph.output_bytes_per_task, j % num_devices);
      }
      matrix[i].data[j].device_id = j % num_devices;
      matrix[i].data[j].output_buff = unique_outputs[i][output_id];
    }
  }

  // Show the information of the task graphs
  app.display();

  double start = omp_get_wtime(); 

  // Run the task graph in parallel
  #pragma omp parallel num_threads(num_devices+1)
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

          // printf("num_args=%d\n", num_args);

          // Calculate the position of the task to execute
          int x0 = args[0].x;
          int y0 = args[0].y;
          int point0 = y0 * graph.max_width + x0;

          // Get the pointer to the task output data
          tile_t *tile_in0 = &mat[point0];
          char *output_ptr = (char*)tile_in0->output_buff;

          // Get the id of the device that will execute the task
          int device_id = tile_in0->device_id;

          unsigned long output_size = graph.output_bytes_per_task;

          // Depending on the number of dependencies
          switch(num_args)
          {
            case 1:
            {
              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])

              #pragma omp target nowait \
                depend(inout: output_ptr) \
                depend(depobj: tile_in0->dep) \
                is_device_ptr(output_ptr) \
                device(device_id)
              {
                char * input_ptrs[num_args];
                input_ptrs[0] = output_ptr;
                task_execute(graph, x0, y0, num_args, input_ptrs, output_ptr);
              }
              break;
            }
            case 2:
            {
              //Get the pointers to the tak input data
              int point1 = args[1].y * graph.max_width+ args[1].x;
              tile_t *tile_in1 = &mat[point1];
              char *input_ptr_in0 = (char*)tile_in1->output_buff;
              char *input_ptr0 = (char*) omp_target_alloc(output_size, device_id);

              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])

              omp_target_memcpy_async(input_ptr0, input_ptr_in0, output_size, 0, 0, device_id, tile_in1->device_id, 1, &tile_in1->dep);

              // Pragma to send the task a node to execute it
              #pragma omp target nowait \
                depend(depobj: tile_in1->dep, tile_in0->dep) \
                is_device_ptr(output_ptr, input_ptr0) \
                device(device_id)
              {
                // Join the input data into a single variable
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                // Execute the task
                task_execute(graph, x0, y0, num_args-1, input_ptrs, output_ptr);
              }

              #pragma omp task depend(depobj: tile_in1->dep)
              omp_target_free(input_ptr0, device_id);
              
              break;
            }
            case 3:
            {
              int point1 = args[1].y * graph.max_width+ args[1].x;
              int point2 = args[2].y * graph.max_width+ args[2].x;

              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];

              char *input_ptr_in0 = (char*)tile_in1->output_buff;
              char *input_ptr_in1 = (char*)tile_in2->output_buff;
              
              char *input_ptr0 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr1 = (char*) omp_target_alloc(output_size, device_id);

              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])

              omp_target_memcpy_async(input_ptr0, input_ptr_in0, output_size, 0, 0, device_id, tile_in1->device_id, 1, &tile_in1->dep);
              omp_target_memcpy_async(input_ptr1, input_ptr_in1, output_size, 0, 0, device_id, tile_in2->device_id, 1, &tile_in2->dep);

              #pragma omp target nowait \
                depend(depobj: tile_in0->dep) \
                depend(depobj: tile_in1->dep, tile_in2->dep) \
                is_device_ptr(output_ptr, input_ptr0, input_ptr1) \
                device(device_id)
              {                 
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                task_execute(graph, x0, y0, num_args-1, input_ptrs, output_ptr);
              }

              #pragma omp task depend(depobj: tile_in1->dep)
              omp_target_free(input_ptr0, device_id);              

              #pragma omp task depend(depobj: tile_in2->dep)
              omp_target_free(input_ptr1, device_id); 

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

              char *input_ptr_in0 = (char*)tile_in1->output_buff;
              char *input_ptr_in1 = (char*)tile_in2->output_buff;
              char *input_ptr_in2 = (char*)tile_in3->output_buff;
              
              char *input_ptr0 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr1 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr2 = (char*) omp_target_alloc(output_size, device_id);
              
              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])
              
              omp_target_memcpy_async(input_ptr0, input_ptr_in0, output_size, 0, 0, device_id, tile_in1->device_id, 1, &tile_in1->dep);
              omp_target_memcpy_async(input_ptr1, input_ptr_in1, output_size, 0, 0, device_id, tile_in2->device_id, 1, &tile_in2->dep);
              omp_target_memcpy_async(input_ptr2, input_ptr_in2, output_size, 0, 0, device_id, tile_in3->device_id, 1, &tile_in3->dep);

              omp_depend_t dep0 = tile_in1->dep;
              omp_depend_t dep1 = tile_in2->dep;
              omp_depend_t dep2 = tile_in3->dep;

              #pragma omp target nowait \
                depend(depobj: tile_in0->dep) \
                depend(depobj: dep0, dep1, dep2) \
                is_device_ptr(output_ptr, input_ptr0, input_ptr1, input_ptr2) \
                device(device_id)
              {                
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                task_execute(graph, x0, y0, num_args-1, input_ptrs, output_ptr);
              }

              #pragma omp task depend(depobj: dep0)
              omp_target_free(input_ptr0, device_id);

              #pragma omp task depend(depobj: dep1)
              omp_target_free(input_ptr1, device_id);

              #pragma omp task depend(depobj: dep2)
              omp_target_free(input_ptr2, device_id);

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

              char *input_ptr_in0 = (char*)tile_in1->output_buff;
              char *input_ptr_in1 = (char*)tile_in2->output_buff;
              char *input_ptr_in2 = (char*)tile_in3->output_buff;
              char *input_ptr_in3 = (char*)tile_in4->output_buff;
              
              char *input_ptr0 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr1 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr2 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr3 = (char*) omp_target_alloc(output_size, device_id);

              omp_depend_t dep0 = tile_in1->dep;
              omp_depend_t dep1 = tile_in2->dep;
              omp_depend_t dep2 = tile_in3->dep;
              omp_depend_t dep3 = tile_in4->dep;

              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])

              omp_target_memcpy_async(input_ptr0, input_ptr_in0, output_size, 0, 0, device_id, tile_in1->device_id, 1, &dep0);
              omp_target_memcpy_async(input_ptr1, input_ptr_in1, output_size, 0, 0, device_id, tile_in2->device_id, 1, &dep1);
              omp_target_memcpy_async(input_ptr2, input_ptr_in2, output_size, 0, 0, device_id, tile_in3->device_id, 1, &dep2);
              omp_target_memcpy_async(input_ptr3, input_ptr_in3, output_size, 0, 0, device_id, tile_in4->device_id, 1, &dep3);

              #pragma omp target nowait \
                depend(depobj: tile_in0->dep) \
                depend(depobj: dep0, dep1, dep2, dep3) \
                is_device_ptr(output_ptr, input_ptr0, input_ptr1, input_ptr2, input_ptr3) \
                device(device_id)
              {
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                input_ptrs[3] = input_ptr3;
                task_execute(graph, x0, y0, num_args-1, input_ptrs, output_ptr);
              }

              #pragma omp task depend(depobj: dep0)
              omp_target_free(input_ptr0, device_id);

              #pragma omp task depend(depobj: dep1)
              omp_target_free(input_ptr1, device_id);

              #pragma omp task depend(depobj: dep2)
              omp_target_free(input_ptr2, device_id);

              #pragma omp task depend(depobj: dep3)
              omp_target_free(input_ptr3, device_id);

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

              char *input_ptr_in0 = (char*)tile_in1->output_buff;
              char *input_ptr_in1 = (char*)tile_in2->output_buff;
              char *input_ptr_in2 = (char*)tile_in3->output_buff;
              char *input_ptr_in3 = (char*)tile_in4->output_buff;
              char *input_ptr_in4 = (char*)tile_in5->output_buff;
              
              char *input_ptr0 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr1 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr2 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr3 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr4 = (char*) omp_target_alloc(output_size, device_id);

              omp_depend_t dep0 = tile_in1->dep;
              omp_depend_t dep1 = tile_in2->dep;
              omp_depend_t dep2 = tile_in3->dep;
              omp_depend_t dep3 = tile_in4->dep;
              omp_depend_t dep4 = tile_in5->dep;

              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])

              omp_target_memcpy_async(input_ptr0, input_ptr_in0, output_size, 0, 0, device_id, tile_in1->device_id, 1, &dep0);
              omp_target_memcpy_async(input_ptr1, input_ptr_in1, output_size, 0, 0, device_id, tile_in2->device_id, 1, &dep1);
              omp_target_memcpy_async(input_ptr2, input_ptr_in2, output_size, 0, 0, device_id, tile_in3->device_id, 1, &dep2);
              omp_target_memcpy_async(input_ptr3, input_ptr_in3, output_size, 0, 0, device_id, tile_in4->device_id, 1, &dep3);
              omp_target_memcpy_async(input_ptr4, input_ptr_in4, output_size, 0, 0, device_id, tile_in5->device_id, 1, &dep4);

              #pragma omp target nowait \
                depend(depobj: tile_in0->dep) \
                depend(depobj: dep0, dep1, dep2, dep3, dep4) \
                is_device_ptr(output_ptr, input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4) \
                device(device_id)
              {           
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                input_ptrs[3] = input_ptr3;
                input_ptrs[4] = input_ptr4;
                task_execute(graph, x0, y0, num_args-1, input_ptrs, output_ptr);
              }

              #pragma omp task depend(depobj: dep0)
              omp_target_free(input_ptr0, device_id);

              #pragma omp task depend(depobj: dep1)
              omp_target_free(input_ptr1, device_id);

              #pragma omp task depend(depobj: dep2)
              omp_target_free(input_ptr2, device_id);

              #pragma omp task depend(depobj: dep3)
              omp_target_free(input_ptr3, device_id);

              #pragma omp task depend(depobj: dep4)
              omp_target_free(input_ptr4, device_id);
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

              char *input_ptr_in0 = (char*)tile_in1->output_buff;
              char *input_ptr_in1 = (char*)tile_in2->output_buff;
              char *input_ptr_in2 = (char*)tile_in3->output_buff;
              char *input_ptr_in3 = (char*)tile_in4->output_buff;
              char *input_ptr_in4 = (char*)tile_in5->output_buff;
              char *input_ptr_in5 = (char*)tile_in6->output_buff;
              
              char *input_ptr0 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr1 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr2 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr3 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr4 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr5 = (char*) omp_target_alloc(output_size, device_id);

              omp_depend_t dep0 = tile_in1->dep;
              omp_depend_t dep1 = tile_in2->dep;
              omp_depend_t dep2 = tile_in3->dep;
              omp_depend_t dep3 = tile_in4->dep;
              omp_depend_t dep4 = tile_in5->dep;
              omp_depend_t dep5 = tile_in6->dep;

              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])

              omp_target_memcpy_async(input_ptr0, input_ptr_in0, output_size, 0, 0, device_id, tile_in1->device_id, 1, &dep0);
              omp_target_memcpy_async(input_ptr1, input_ptr_in1, output_size, 0, 0, device_id, tile_in2->device_id, 1, &dep1);
              omp_target_memcpy_async(input_ptr2, input_ptr_in2, output_size, 0, 0, device_id, tile_in3->device_id, 1, &dep2);
              omp_target_memcpy_async(input_ptr3, input_ptr_in3, output_size, 0, 0, device_id, tile_in4->device_id, 1, &dep3);
              omp_target_memcpy_async(input_ptr4, input_ptr_in4, output_size, 0, 0, device_id, tile_in5->device_id, 1, &dep4);
              omp_target_memcpy_async(input_ptr5, input_ptr_in5, output_size, 0, 0, device_id, tile_in6->device_id, 1, &dep5);

              #pragma omp target nowait \
                depend(depobj: tile_in0->dep) \
                depend(depobj: dep0, dep1, dep2, dep3, dep4, dep5) \
                is_device_ptr(output_ptr, input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5) \
                device(device_id)
              {           
                char * input_ptrs[num_args-1];
                input_ptrs[0] = input_ptr0;
                input_ptrs[1] = input_ptr1;
                input_ptrs[2] = input_ptr2;
                input_ptrs[3] = input_ptr3;
                input_ptrs[4] = input_ptr4;
                input_ptrs[5] = input_ptr5;
                task_execute(graph, x0, y0, num_args-1, input_ptrs, output_ptr);
              }

              #pragma omp task depend(depobj: dep0)
              omp_target_free(input_ptr0, device_id);

              #pragma omp task depend(depobj: dep1)
              omp_target_free(input_ptr1, device_id);

              #pragma omp task depend(depobj: dep2)
              omp_target_free(input_ptr2, device_id);

              #pragma omp task depend(depobj: dep3)
              omp_target_free(input_ptr3, device_id);

              #pragma omp task depend(depobj: dep4)
              omp_target_free(input_ptr4, device_id);

              #pragma omp task depend(depobj: dep5)
              omp_target_free(input_ptr5, device_id);
    
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

              char *input_ptr_in0 = (char*)tile_in1->output_buff;
              char *input_ptr_in1 = (char*)tile_in2->output_buff;
              char *input_ptr_in2 = (char*)tile_in3->output_buff;
              char *input_ptr_in3 = (char*)tile_in4->output_buff;
              char *input_ptr_in4 = (char*)tile_in5->output_buff;
              char *input_ptr_in5 = (char*)tile_in6->output_buff;
              char *input_ptr_in6 = (char*)tile_in7->output_buff;
              
              char *input_ptr0 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr1 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr2 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr3 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr4 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr5 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr6 = (char*) omp_target_alloc(output_size, device_id);

              omp_depend_t dep0 = tile_in1->dep;
              omp_depend_t dep1 = tile_in2->dep;
              omp_depend_t dep2 = tile_in3->dep;
              omp_depend_t dep3 = tile_in4->dep;
              omp_depend_t dep4 = tile_in5->dep;
              omp_depend_t dep5 = tile_in6->dep;
              omp_depend_t dep6 = tile_in7->dep;

              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])

              omp_target_memcpy_async(input_ptr0, input_ptr_in0, output_size, 0, 0, device_id, tile_in1->device_id, 1, &dep0);
              omp_target_memcpy_async(input_ptr1, input_ptr_in1, output_size, 0, 0, device_id, tile_in2->device_id, 1, &dep1);
              omp_target_memcpy_async(input_ptr2, input_ptr_in2, output_size, 0, 0, device_id, tile_in3->device_id, 1, &dep2);
              omp_target_memcpy_async(input_ptr3, input_ptr_in3, output_size, 0, 0, device_id, tile_in4->device_id, 1, &dep3);
              omp_target_memcpy_async(input_ptr4, input_ptr_in4, output_size, 0, 0, device_id, tile_in5->device_id, 1, &dep4);
              omp_target_memcpy_async(input_ptr5, input_ptr_in5, output_size, 0, 0, device_id, tile_in6->device_id, 1, &dep5);
              omp_target_memcpy_async(input_ptr6, input_ptr_in6, output_size, 0, 0, device_id, tile_in7->device_id, 1, &dep6);

              #pragma omp target nowait \
                depend(depobj: tile_in0->dep) \
                depend(depobj: dep0, dep1, dep2, dep3, dep4, dep5, dep6) \
                is_device_ptr(output_ptr, input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6) \
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
                task_execute(graph, x0, y0, num_args-1, input_ptrs, output_ptr);
              }

              #pragma omp task depend(depobj: dep0)
              omp_target_free(input_ptr0, device_id);

              #pragma omp task depend(depobj: dep1)
              omp_target_free(input_ptr1, device_id);

              #pragma omp task depend(depobj: dep2)
              omp_target_free(input_ptr2, device_id);

              #pragma omp task depend(depobj: dep3)
              omp_target_free(input_ptr3, device_id);

              #pragma omp task depend(depobj: dep4)
              omp_target_free(input_ptr4, device_id);

              #pragma omp task depend(depobj: dep5)
              omp_target_free(input_ptr5, device_id);

              #pragma omp task depend(depobj: dep6)
              omp_target_free(input_ptr6, device_id);
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

              char *input_ptr_in0 = (char*)tile_in1->output_buff;
              char *input_ptr_in1 = (char*)tile_in2->output_buff;
              char *input_ptr_in2 = (char*)tile_in3->output_buff;
              char *input_ptr_in3 = (char*)tile_in4->output_buff;
              char *input_ptr_in4 = (char*)tile_in5->output_buff;
              char *input_ptr_in5 = (char*)tile_in6->output_buff;
              char *input_ptr_in6 = (char*)tile_in7->output_buff;
              char *input_ptr_in7 = (char*)tile_in8->output_buff;
              
              char *input_ptr0 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr1 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr2 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr3 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr4 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr5 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr6 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr7 = (char*) omp_target_alloc(output_size, device_id);

              omp_depend_t dep0 = tile_in1->dep;
              omp_depend_t dep1 = tile_in2->dep;
              omp_depend_t dep2 = tile_in3->dep;
              omp_depend_t dep3 = tile_in4->dep;
              omp_depend_t dep4 = tile_in5->dep;
              omp_depend_t dep5 = tile_in6->dep;
              omp_depend_t dep6 = tile_in7->dep;
              omp_depend_t dep7 = tile_in8->dep;

              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])

              omp_target_memcpy_async(input_ptr0, input_ptr_in0, output_size, 0, 0, device_id, tile_in1->device_id, 1, &dep0);
              omp_target_memcpy_async(input_ptr1, input_ptr_in1, output_size, 0, 0, device_id, tile_in2->device_id, 1, &dep1);
              omp_target_memcpy_async(input_ptr2, input_ptr_in2, output_size, 0, 0, device_id, tile_in3->device_id, 1, &dep2);
              omp_target_memcpy_async(input_ptr3, input_ptr_in3, output_size, 0, 0, device_id, tile_in4->device_id, 1, &dep3);
              omp_target_memcpy_async(input_ptr4, input_ptr_in4, output_size, 0, 0, device_id, tile_in5->device_id, 1, &dep4);
              omp_target_memcpy_async(input_ptr5, input_ptr_in5, output_size, 0, 0, device_id, tile_in6->device_id, 1, &dep5);
              omp_target_memcpy_async(input_ptr6, input_ptr_in6, output_size, 0, 0, device_id, tile_in7->device_id, 1, &dep6);
              omp_target_memcpy_async(input_ptr7, input_ptr_in7, output_size, 0, 0, device_id, tile_in8->device_id, 1, &dep7);

              #pragma omp target nowait \
                depend(depobj: tile_in0->dep) \
                depend(depobj: dep0, dep1, dep2, dep3, dep4, dep5, dep6, dep7) \
                is_device_ptr(output_ptr, input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7) \
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
                task_execute(graph, x0, y0, num_args-1, input_ptrs, output_ptr);
              }

              #pragma omp task depend(depobj: dep0)
              omp_target_free(input_ptr0, device_id);

              #pragma omp task depend(depobj: dep1)
              omp_target_free(input_ptr1, device_id);

              #pragma omp task depend(depobj: dep2)
              omp_target_free(input_ptr2, device_id);

              #pragma omp task depend(depobj: dep3)
              omp_target_free(input_ptr3, device_id);

              #pragma omp task depend(depobj: dep4)
              omp_target_free(input_ptr4, device_id);

              #pragma omp task depend(depobj: dep5)
              omp_target_free(input_ptr5, device_id);

              #pragma omp task depend(depobj: dep6)
              omp_target_free(input_ptr6, device_id);

              #pragma omp task depend(depobj: dep7)
              omp_target_free(input_ptr7, device_id);

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

              char *input_ptr_in0 = (char*)tile_in1->output_buff;
              char *input_ptr_in1 = (char*)tile_in2->output_buff;
              char *input_ptr_in2 = (char*)tile_in3->output_buff;
              char *input_ptr_in3 = (char*)tile_in4->output_buff;
              char *input_ptr_in4 = (char*)tile_in5->output_buff;
              char *input_ptr_in5 = (char*)tile_in6->output_buff;
              char *input_ptr_in6 = (char*)tile_in7->output_buff;
              char *input_ptr_in7 = (char*)tile_in8->output_buff;
              char *input_ptr_in8 = (char*)tile_in9->output_buff;

              char *input_ptr0 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr1 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr2 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr3 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr4 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr5 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr6 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr7 = (char*) omp_target_alloc(output_size, device_id);
              char *input_ptr8 = (char*) omp_target_alloc(output_size, device_id);

              omp_depend_t dep0 = tile_in1->dep;
              omp_depend_t dep1 = tile_in2->dep;
              omp_depend_t dep2 = tile_in3->dep;
              omp_depend_t dep3 = tile_in4->dep;
              omp_depend_t dep4 = tile_in5->dep;
              omp_depend_t dep5 = tile_in6->dep;
              omp_depend_t dep6 = tile_in7->dep;
              omp_depend_t dep7 = tile_in8->dep;
              omp_depend_t dep8 = tile_in9->dep;

              #pragma omp depobj(tile_in0->dep) depend(inout: output_ptr[0:output_size])

              omp_target_memcpy_async(input_ptr0, input_ptr_in0, output_size, 0, 0, device_id, tile_in1->device_id, 1, &dep0);
              omp_target_memcpy_async(input_ptr1, input_ptr_in1, output_size, 0, 0, device_id, tile_in2->device_id, 1, &dep1);
              omp_target_memcpy_async(input_ptr2, input_ptr_in2, output_size, 0, 0, device_id, tile_in3->device_id, 1, &dep2);
              omp_target_memcpy_async(input_ptr3, input_ptr_in3, output_size, 0, 0, device_id, tile_in4->device_id, 1, &dep3);
              omp_target_memcpy_async(input_ptr4, input_ptr_in4, output_size, 0, 0, device_id, tile_in5->device_id, 1, &dep4);
              omp_target_memcpy_async(input_ptr5, input_ptr_in5, output_size, 0, 0, device_id, tile_in6->device_id, 1, &dep5);
              omp_target_memcpy_async(input_ptr6, input_ptr_in6, output_size, 0, 0, device_id, tile_in7->device_id, 1, &dep6);
              omp_target_memcpy_async(input_ptr7, input_ptr_in7, output_size, 0, 0, device_id, tile_in8->device_id, 1, &dep7);
              omp_target_memcpy_async(input_ptr8, input_ptr_in8, output_size, 0, 0, device_id, tile_in9->device_id, 1, &dep8);

              #pragma omp target nowait \
                depend(depobj: tile_in0->dep) \
                depend(depobj: dep0, dep1, dep2, dep3, dep4, dep5, dep6, dep7, dep8) \
                is_device_ptr(output_ptr, input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7, input_ptr8) \
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
                task_execute(graph, x0, y0, num_args-1, input_ptrs, output_ptr);
              }

              #pragma omp task depend(depobj: dep0)
              omp_target_free(input_ptr0, device_id);

              #pragma omp task depend(depobj: dep1)
              omp_target_free(input_ptr1, device_id);

              #pragma omp task depend(depobj: dep2)
              omp_target_free(input_ptr2, device_id);

              #pragma omp task depend(depobj: dep3)
              omp_target_free(input_ptr3, device_id);

              #pragma omp task depend(depobj: dep4)
              omp_target_free(input_ptr4, device_id);

              #pragma omp task depend(depobj: dep5)
              omp_target_free(input_ptr5, device_id);

              #pragma omp task depend(depobj: dep6)
              omp_target_free(input_ptr6, device_id);

              #pragma omp task depend(depobj: dep7)
              omp_target_free(input_ptr7, device_id);

              #pragma omp task depend(depobj: dep8)
              omp_target_free(input_ptr8, device_id);

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
  app.report_timing(elapsed);

  // Deallocate memory block used to storethe information of the Task Graph
  for (unsigned i = 0; i < graphs.size(); i++) {
    // Minimum number of unique buffers needed to execute a task (conservative choice)
    int num_outputs = 2 * graphs[i].max_width;
    int device_id, output_id;
    for (int j = 0; j < num_outputs; j++)
      omp_target_free(matrix[i].data[j].output_buff, matrix[i].data[j].device_id);

    // #pragma omp parallel for num_threads(num_devices)
    // for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
    //   omp_target_free(matrix[i].data[j].output_buff, matrix[i].data[j].device_id);
    //   matrix[i].data[j].output_buff = NULL;
    // }
    free(matrix[i].data);
    // free(unique_outputs[i]);
    matrix[i].data = NULL;
  }
  
  free(matrix);
  // free(unique_outputs);
  matrix = NULL;
  return 0;
}