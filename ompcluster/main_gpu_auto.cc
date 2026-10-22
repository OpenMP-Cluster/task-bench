// C headers
#include <assert.h>
#include <omp.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

// C++ headers
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <memory>

// Project headers
#include "core.h"
#include "timer.h"
//#include "kmp.h"

typedef int kmp_int32;
struct ident_t;
extern "C"{
 kmp_int32 __kmpc_start_record_task(ident_t *loc, kmp_int32 gtid,
                                              kmp_int32 input_flags,
                                              kmp_int32 tdg_id);
 void __kmpc_end_record_task(ident_t *loc, kmp_int32 gtid,
                                       kmp_int32 input_flags, kmp_int32 tdg_id);

 void __kmpc_dmdas_add_next_task_perf_feature(
     int kind,
     double value);

 void __kmpc_dmdas_save_perf_model();
}

enum {
  KMP_DMDAS_FEATURE_DATA_BYTES = 0,
  KMP_DMDAS_FEATURE_ITERATIONS = 1,
  KMP_DMDAS_FEATURE_FLOPS = 2,
  KMP_DMDAS_FEATURE_USER = 100
};
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


#define UNROLL_16 16

void task_execute_gpu(long iter, double *A, unsigned long N,
                      int block_size, int num_blocks)
{
  #pragma omp distribute parallel for
  for (int tid = 0; tid < block_size * num_blocks; tid++) {
    double A_local[UNROLL_16];
    int total_threads = block_size * num_blocks;

    #pragma unroll
    for (int u = 0; u < UNROLL_16; u++) {
      int idx = tid + u * total_threads;
      if (idx < N)
        A_local[u] = A[idx];
      else
        A_local[u] = 1.2345;
    }

    for (long i = 0; i < iter; i++) {
      #pragma unroll
      for (int u = 0; u < UNROLL_16; u++)
        A_local[u] = fma(A_local[u], A_local[u], A_local[u]);
    }

    #pragma unroll
    for (int u = 0; u < UNROLL_16; u++) {
      int idx = tid + u * total_threads;
      if (idx < N)
        A[idx] = A_local[u];
    }
  }
}

void task_execute_memory(long iter, double *A, unsigned long N,
                         int block_size, int num_blocks)
{
  int total_threads = block_size * num_blocks;

  // Initialize the buffer deterministically.
  #pragma omp distribute parallel for
  for (unsigned long idx = 0; idx < N; idx++) {
    A[idx] = static_cast<double>(idx & 0xff);
  }

  // Memory-intensive phase: repeatedly read and write the whole buffer.
  for (long pass = 0; pass < iter; pass++) {
    #pragma omp distribute parallel for
    for (unsigned long idx = 0; idx < N; idx++) {
      A[idx] = A[idx] + 1.0;
    }
  }
}

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

  // Setting up the number of iteration of the taskgraph, default : 5
  int loop_number = 5;
  if(char *env_str = std::getenv("OMPCLUSTER_TASKBENCH_ITERATION_NUMBER")) {
    loop_number = std::stoi(env_str);
  }

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

  TaskGraph* graphs_array = graphs.data();

  // Show the information of the task graphs
  new_app.display();

  double start = omp_get_wtime();

  #pragma omp parallel
  #pragma omp master
  {
    for(int loop = 0 ; loop < loop_number ; loop ++) {
      double start_loop = omp_get_wtime();
    int res = __kmpc_start_record_task(nullptr, 0, 0,0);
    if(res){
  // Transfer the graph data
    #pragma omp target enter data nowait                                       \
        map(to: graphs_array[:graphs.size()])                                  \
        depend(out: *graphs_array)

    // Run the task graph in parallel
    for (unsigned graph_idx = 0; graph_idx < graphs.size(); graph_idx++) {
      const TaskGraph &graph = graphs[graph_idx];
      tile_t *mat = matrix[graph_idx].data;
      size_t graph_out = graph.output_bytes_per_task;

      // Alloc the data of each task on-demand using the memory management
      // (asynchronous data maps).
      int number_point = matrix[graph_idx].M * matrix[graph_idx].N;
      for (size_t i = 0; i < number_point; i++) {
        tile_t *tile_in = &mat[i];
        char *output_ptr = (char*)tile_in->output_buff;
        #pragma omp target enter data nowait                                   \
            map(alloc: output_ptr[:graph_out])                                 \
            depend(out: *output_ptr)
      }

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

          long iter = graph.kernel.iterations;

          long compute_iter = iter;
          long memory_iter = iter;

          if (char *env_str =
                  std::getenv("OMPCLUSTER_TASKBENCH_COMPUTE_ITER")) {
            compute_iter = std::stol(env_str);
          }

          if (char *env_str =
                  std::getenv("OMPCLUSTER_TASKBENCH_MEMORY_ITER")) {
            memory_iter = std::stol(env_str);
          }

          unsigned long N = graph.output_bytes_per_task / sizeof(double);
          int block_size = graph.tpblock;
          int num_blocks = graph.nblocks;


          // Heterogeneous workload class and effective work parameter.
          const bool is_compute = ((x0 + y0) % 2) == 0;
          const long effective_iter =
              is_compute ? compute_iter : memory_iter;

          // Attach the workload parameter to the next OpenMP target task.
          // DMDAS consumes these pending features when that task is created.
          __kmpc_dmdas_add_next_task_perf_feature(
              KMP_DMDAS_FEATURE_ITERATIONS,
              static_cast<double>(effective_iter));

          // Optional diagnostic for heterogeneous workload assignment.
          // Enabled only when OMPCLUSTER_TASKBENCH_HET_DEBUG is set.
          if (std::getenv("OMPCLUSTER_TASKBENCH_HET_DEBUG")) {
            fprintf(stderr,
                    "[TB-HET] x=%d y=%d deps=%d kind=%s "
                    "graph_iter=%ld effective_iter=%ld "
                    "elements=%lu tpblock=%d nblocks=%d\n",
                    x0, y0, num_args,
                    is_compute ? "compute" : "memory",
                    iter,
                    effective_iter,
                    N, block_size, num_blocks);
          }

          // Depending on the number of dependencies
          switch(num_args)
          {
            case 1:
            {
              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) \
                  nowait                                                               \
                  depend(in: *graphs_array)                                            \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) \
                  nowait                                                               \
                  depend(in: *graphs_array)                                            \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
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
              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array)                                                    \
                  depend(in: *input_ptr0)                                                      \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array)                                                    \
                  depend(in: *input_ptr0)                                                      \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
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

              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array)                                                    \
                  depend(in: *input_ptr0)                                                      \
                  depend(in: *input_ptr1)                                                      \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array)                                                    \
                  depend(in: *input_ptr0)                                                      \
                  depend(in: *input_ptr1)                                                      \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
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

              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array)                                                    \
                  depend(in: *input_ptr0)                                                      \
                  depend(in: *input_ptr1)                                                      \
                  depend(in: *input_ptr2)                                                      \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array)                                                    \
                  depend(in: *input_ptr0)                                                      \
                  depend(in: *input_ptr1)                                                      \
                  depend(in: *input_ptr2)                                                      \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
              }
              break;
            }
            case 5:
            {
              int point1 = args[1].y * graph.max_width + args[1].x;
              int point2 = args[2].y * graph.max_width + args[2].x;
              int point3 = args[3].y * graph.max_width + args[3].x;
              int point4 = args[4].y * graph.max_width + args[4].x;

              tile_t *tile_in1 = &mat[point1];
              tile_t *tile_in2 = &mat[point2];
              tile_t *tile_in3 = &mat[point3];
              tile_t *tile_in4 = &mat[point4];

              char *input_ptr0 = (char*)tile_in1->output_buff;
              char *input_ptr1 = (char*)tile_in2->output_buff;
              char *input_ptr2 = (char*)tile_in3->output_buff;
              char *input_ptr3 = (char*)tile_in4->output_buff;

              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
              }
              break;
            }

            case 6:
            {
              int point1 = args[1].y * graph.max_width + args[1].x;
              int point2 = args[2].y * graph.max_width + args[2].x;
              int point3 = args[3].y * graph.max_width + args[3].x;
              int point4 = args[4].y * graph.max_width + args[4].x;
              int point5 = args[5].y * graph.max_width + args[5].x;

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

              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
              }
              break;
            }

            case 7:
            {
              int point1 = args[1].y * graph.max_width + args[1].x;
              int point2 = args[2].y * graph.max_width + args[2].x;
              int point3 = args[3].y * graph.max_width + args[3].x;
              int point4 = args[4].y * graph.max_width + args[4].x;
              int point5 = args[5].y * graph.max_width + args[5].x;
              int point6 = args[6].y * graph.max_width + args[6].x;

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

              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(in: *input_ptr5) \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(in: *input_ptr5) \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
              }
              break;
            }

            case 8:
            {
              int point1 = args[1].y * graph.max_width + args[1].x;
              int point2 = args[2].y * graph.max_width + args[2].x;
              int point3 = args[3].y * graph.max_width + args[3].x;
              int point4 = args[4].y * graph.max_width + args[4].x;
              int point5 = args[5].y * graph.max_width + args[5].x;
              int point6 = args[6].y * graph.max_width + args[6].x;
              int point7 = args[7].y * graph.max_width + args[7].x;

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

              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(in: *input_ptr5) \
                  depend(in: *input_ptr6) \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(in: *input_ptr5) \
                  depend(in: *input_ptr6) \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
              }
              break;
            }

            case 9:
            {
              int point1 = args[1].y * graph.max_width + args[1].x;
              int point2 = args[2].y * graph.max_width + args[2].x;
              int point3 = args[3].y * graph.max_width + args[3].x;
              int point4 = args[4].y * graph.max_width + args[4].x;
              int point5 = args[5].y * graph.max_width + args[5].x;
              int point6 = args[6].y * graph.max_width + args[6].x;
              int point7 = args[7].y * graph.max_width + args[7].x;
              int point8 = args[8].y * graph.max_width + args[8].x;

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

              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(in: *input_ptr5) \
                  depend(in: *input_ptr6) \
                  depend(in: *input_ptr7) \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(in: *input_ptr5) \
                  depend(in: *input_ptr6) \
                  depend(in: *input_ptr7) \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
              }
              break;
            }

            case 10:
            {
              int point1 = args[1].y * graph.max_width + args[1].x;
              int point2 = args[2].y * graph.max_width + args[2].x;
              int point3 = args[3].y * graph.max_width + args[3].x;
              int point4 = args[4].y * graph.max_width + args[4].x;
              int point5 = args[5].y * graph.max_width + args[5].x;
              int point6 = args[6].y * graph.max_width + args[6].x;
              int point7 = args[7].y * graph.max_width + args[7].x;
              int point8 = args[8].y * graph.max_width + args[8].x;
              int point9 = args[9].y * graph.max_width + args[9].x;

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

              if (((x0 + y0) % 2) == 0) {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(in: *input_ptr5) \
                  depend(in: *input_ptr6) \
                  depend(in: *input_ptr7) \
                  depend(in: *input_ptr8) \
                  depend(inout: *output_ptr)
                {
                  task_execute_gpu(compute_iter, (double *)output_ptr, N,
                                  block_size, num_blocks);
                }
              } else {
                #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) nowait \
                  depend(in: *graphs_array) \
                  depend(in: *input_ptr0) \
                  depend(in: *input_ptr1) \
                  depend(in: *input_ptr2) \
                  depend(in: *input_ptr3) \
                  depend(in: *input_ptr4) \
                  depend(in: *input_ptr5) \
                  depend(in: *input_ptr6) \
                  depend(in: *input_ptr7) \
                  depend(in: *input_ptr8) \
                  depend(inout: *output_ptr)
                {
                  task_execute_memory(memory_iter, (double *)output_ptr, N,
                                      block_size, num_blocks);
                }
              }
              break;
            }
              default:
                assert(false && "unexpected num_args");
            }
        }
      }

      // Delete the data of each task on-demand using (asynchronous data maps)
      // after their respective tasks have used them.
      for (size_t i = 0; i < number_point; i++) {
        tile_t *tile_in = &mat[i];
        char *output_ptr = (char *)tile_in->output_buff;
        #pragma omp target exit data nowait                                    \
                map(release: output_ptr[:graph_out])                           \
                depend(inout: *output_ptr)
      }
    }

    #pragma omp target exit data nowait                                        \
              map(release: graphs_array[:graphs.size()])                       \
              depend(inout: *graphs_array)
  }
    __kmpc_end_record_task(nullptr,0,0,0);

  double end_loop = omp_get_wtime();
  double elapsed_loop = end_loop - start_loop;
  printf("Loop %d : Elapsed time : %f\n",loop, elapsed_loop);
    }
  }
  // Persist the DMDAS performance model after the training execution.
  __kmpc_dmdas_save_perf_model();

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
