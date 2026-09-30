/* Copyright 2020 Los Alamos National Laboratory
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>
#include <array>
#include <starpu_mpi.h>
#include <starpu_profiling.h>
#include <starpu_worker.h>
#include <omp.h>
#include "data.h"
#include "core.h"
#include "timer.h"

#include <unistd.h>

#define VERBOSE_LEVEL 0

#define ENABLE_PRUNE_MPI_TASK_INSERT

#define UNROLL_16 16

typedef struct payload_s {
  int graph_id;
  int i;
  int j;
  const TaskGraph *graph;
}payload_t;

void task_execute_gpu(long iter, double *A, unsigned long N, int block_size, int num_blocks)
{
  #pragma omp distribute parallel for
  for(int tid = 0; tid < block_size * num_blocks; tid++) {
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

    for(long i=0; i < iter; i++) {
      #pragma unroll
      for (int u = 0; u < UNROLL_16; u++) {
        A_local[u] = fma(A_local[u], A_local[u], A_local[u]);
      }
    }

    #pragma unroll
    for (int u = 0; u < UNROLL_16; u++) {
      int idx = tid + u * total_threads;
      if (idx < N)
        A[idx] = A_local[u];
    }
  }
}

static void launch_gpu_kernel_hip(const TaskGraph *graph, const payload_t &payload, char *output_ptr)
{
  long iter = graph->kernel.iterations;
  unsigned long N = graph->output_bytes_per_task/sizeof(double);
  int block_size = graph->tpblock;
  int num_blocks = graph->nblocks;

  // Run the OpenMP target region on the GPU bound to the StarPU HIP worker
  // executing this codelet. output_ptr is a StarPU device pointer, so the
  // is_device_ptr clause skips OpenMP map/copy overhead entirely.
  int dev_id = starpu_worker_get_devid(starpu_worker_get_id());

  #pragma omp target teams num_teams(num_blocks) thread_limit(block_size) \
          is_device_ptr(output_ptr) device(dev_id)
  {
    task_execute_gpu(iter, (double *)output_ptr, N, block_size, num_blocks);
  }
}

static void task1(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

static void task2(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[1]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

static void task3(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[2]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

static void task4(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[3]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

static void task5(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[4]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

static void task6(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[5]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

static void task7(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[6]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

static void task8(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[7]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

static void task9(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[8]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

static void task10(void *descr[], void *cl_arg)
{
  float *out;
  payload_t payload;
  out = (float *)STARPU_MATRIX_GET_PTR(descr[9]);
  starpu_codelet_unpack_args(cl_arg, &payload);

  launch_gpu_kernel_hip(payload.graph, payload, (char*)out);
}

struct starpu_codelet cl_task1; 
struct starpu_codelet cl_task2;
struct starpu_codelet cl_task3;
struct starpu_codelet cl_task4;
struct starpu_codelet cl_task5;
struct starpu_codelet cl_task6;
struct starpu_codelet cl_task7;
struct starpu_codelet cl_task8;
struct starpu_codelet cl_task9;
struct starpu_codelet cl_task10;

typedef struct matrix_s {
  int MT;
  int NT;
  starpu_ddesc_t *ddescA;
}matrix_t;

struct StarPUApp : public App {
public:
  StarPUApp(int argc, char **argv);
  ~StarPUApp();
  void execute_main_loop();
  void execute_timestep(size_t idx, long t);
private:
  void insert_task(int num_args, payload_t &payload, std::array<starpu_data_handle_t, 10> &args);
  void parse_argument(int argc, char **argv);
  void debug_printf(int verbose_level, const char *format, ...);
private:
  struct starpu_conf *conf;
  int rank;
  int world;
  int nb_cores;
  int P;
  int Q;
  int MB;
  matrix_t mat_array[10];
};

void StarPUApp::insert_task(int num_args, payload_t &payload, std::array<starpu_data_handle_t, 10> &args)
{
  void (*callback)(void*) = NULL;
  starpu_ddesc_t *descA = mat_array[payload.graph_id].ddescA;
  switch(num_args) {
  case 1:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task1),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_RW, args[0],
        STARPU_NAME, "task1",
        0);
    break;
  case 2:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task2),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_R, args[1],
        STARPU_RW, args[0],
        STARPU_NAME, "task2",
        0);
    break;
  case 3:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task3),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_R, args[1],
        STARPU_R, args[2],
        STARPU_RW, args[0],
        STARPU_NAME, "task3",
        0);
    break;
  case 4:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task4),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_R, args[1],
        STARPU_R, args[2],
        STARPU_R, args[3],
        STARPU_RW, args[0],
        STARPU_NAME, "task4",
        0);
    break;
  case 5:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task5),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_R, args[1],
        STARPU_R, args[2],
        STARPU_R, args[3],
        STARPU_R, args[4],
        STARPU_RW, args[0],
        STARPU_NAME, "task5",
        0);
    break;
  case 6:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task6),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_R, args[1],
        STARPU_R, args[2],
        STARPU_R, args[3],
        STARPU_R, args[4],
        STARPU_R, args[5],
        STARPU_RW, args[0],
        STARPU_NAME, "task6",
        0);
    break;
  case 7:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task7),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_R, args[1],
        STARPU_R, args[2],
        STARPU_R, args[3],
        STARPU_R, args[4],
        STARPU_R, args[5],
        STARPU_R, args[6],
        STARPU_RW, args[0],
        STARPU_NAME, "task7",
        0);
    break;
  case 8:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task8),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_R, args[1],
        STARPU_R, args[2],
        STARPU_R, args[3],
        STARPU_R, args[4],
        STARPU_R, args[5],
        STARPU_R, args[6],
        STARPU_R, args[7],
        STARPU_RW, args[0],
        STARPU_NAME, "task8",
        0);
    break;
  case 9:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task9),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_R, args[1],
        STARPU_R, args[2],
        STARPU_R, args[3],
        STARPU_R, args[4],
        STARPU_R, args[5],
        STARPU_R, args[6],
        STARPU_R, args[7],
        STARPU_R, args[8],
        STARPU_RW, args[0],
        STARPU_NAME, "task9",
        0);
    break;
  case 10:
    starpu_mpi_insert_task(
        MPI_COMM_WORLD, &(cl_task10),
        STARPU_VALUE,    &payload, sizeof(payload_t),
        STARPU_R, args[1],
        STARPU_R, args[2],
        STARPU_R, args[3],
        STARPU_R, args[4],
        STARPU_R, args[5],
        STARPU_R, args[6],
        STARPU_R, args[7],
        STARPU_R, args[8],
        STARPU_R, args[9],
        STARPU_RW, args[0],
        STARPU_NAME, "task10",
        0);
    break;
  default:
    assert(false && "unexpected num_args");
  };
}

void StarPUApp::parse_argument(int argc, char **argv)
{
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-mb")) {
      MB = atoi(argv[++i]);
    }
    if (!strcmp(argv[i], "-core")) {
      nb_cores = atoi(argv[++i]);
    }
    if (!strcmp(argv[i], "-p")) {
      P = atoi(argv[++i]);
    }
    if (!strcmp(argv[i], "-S")) {
      starpu_enable_supertiling = true;
    }
  }
}

StarPUApp::StarPUApp(int argc, char **argv)
  : App(argc, argv)
{
  cl_task1.where     = STARPU_HIP;                                   
  cl_task1.hip_funcs[0] = task1;                                       
  cl_task1.nbuffers  = 1;                                           
  cl_task1.name      = "task1";
  
  cl_task2.where     = STARPU_HIP;                                   
  cl_task2.hip_funcs[0] = task2;                                       
  cl_task2.nbuffers  = 2;                                           
  cl_task2.name      = "task2";
  
  cl_task3.where     = STARPU_HIP;                                   
  cl_task3.hip_funcs[0] = task3;                                       
  cl_task3.nbuffers  = 3;                                           
  cl_task3.name      = "task3";
  
  cl_task4.where     = STARPU_HIP;                                   
  cl_task4.hip_funcs[0] = task4;                                       
  cl_task4.nbuffers  = 4;                                           
  cl_task4.name      = "task4";
  
  cl_task5.where     = STARPU_HIP;                                   
  cl_task5.hip_funcs[0] = task5;                                       
  cl_task5.nbuffers  = 5;                                           
  cl_task5.name      = "task5";
  
  cl_task6.where     = STARPU_HIP;                                   
  cl_task6.hip_funcs[0] = task6;                                       
  cl_task6.nbuffers  = 6;                                           
  cl_task6.name      = "task6";
  
  cl_task7.where     = STARPU_HIP;                                   
  cl_task7.hip_funcs[0] = task7;                                       
  cl_task7.nbuffers  = 7;                                           
  cl_task7.name      = "task7";
  
  cl_task8.where     = STARPU_HIP;                                   
  cl_task8.hip_funcs[0] = task8;                                       
  cl_task8.nbuffers  = 8;                                           
  cl_task8.name      = "task8";
  
  cl_task9.where     = STARPU_HIP;                                   
  cl_task9.hip_funcs[0] = task9;                                       
  cl_task9.nbuffers  = 9;                                           
  cl_task9.name      = "task9";
  
  cl_task10.where     = STARPU_HIP;                                   
  cl_task10.hip_funcs[0] = task10;                                       
  cl_task10.nbuffers  = 10;                                           
  cl_task10.name      = "task10";
  
  int i;
  
  P = 1;
  MB = 2;
  nb_cores = 1;
  
  parse_argument(argc, argv);
  
  conf =  (struct starpu_conf *)malloc (sizeof(struct starpu_conf));
  starpu_conf_init( conf );

  conf->ncpus = nb_cores;
  conf->ncuda = 0;
  conf->nopencl = 0;
  // One HIP worker per OpenMP target device so StarPU schedules codelets
  // across every GPU, each launching its kernel on the worker-bound device.
  conf->nhip = omp_get_num_devices();
  if (conf->nhip == 0) {
    fprintf(stderr, "No OpenMP target devices found; a GPU is required.\n");
    exit(1);
  }
  conf->sched_policy_name = "dmdar";
  
  int ret;
  ret = starpu_init(conf);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
  ret = starpu_mpi_init(&argc, &argv, 1);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
  starpu_mpi_comm_size(MPI_COMM_WORLD, &world);
  
  Q = world/P;
  assert(P*Q == world);  
  
  int MB_cal = 0;
  
  for (i = 0; i < graphs.size(); i++) {
    TaskGraph &graph = graphs[i];
    matrix_t &mat = mat_array[i];
    
    MB_cal = ceil(sqrt(graph.output_bytes_per_task / sizeof(float)));
    if (MB_cal > MB) {
      MB = MB_cal;
    }
    
    mat.NT = graph.max_width;
    mat.MT = graph.nb_fields;
  
    debug_printf(0, "mb %d, mt %d, nt %d, timesteps %d, enable_supertiling %d, nb_fields %d\n", MB, mat.MT, mat.NT, graph.timesteps, starpu_enable_supertiling, graph.nb_fields);
    assert (graph.output_bytes_per_task <= sizeof(float) * MB * MB);

    mat.ddescA = create_and_distribute_data(rank, world, MB, MB, mat.MT, mat.NT, P, Q, i);
  }
  
  debug_printf(0, "num_gpu_devices %d\n", omp_get_num_devices());
}

StarPUApp::~StarPUApp()
{
  int i;
  for (i = 0; i < graphs.size(); i++) {
    matrix_t &mat = mat_array[i];
    destroy_data(mat.ddescA);
  }  
  if (conf != NULL) {
    free (conf);
  } 
  starpu_mpi_shutdown();
  starpu_shutdown();
}

void StarPUApp::execute_main_loop()
{
  if (rank == 0) {
    display();
  }

  void (*callback)(void*) = NULL;

  int x, y;

  /* Initialize data structures before measurement */
  for (int i = 0; i < graphs.size(); i++) {
    const TaskGraph &g = graphs[i];

    for (y = 0; y < g.timesteps; y++) {
      long offset = g.offset_at_timestep(y);
      long width = g.width_at_timestep(y);
      matrix_t &mat = mat_array[i];
      int nb_fields = g.nb_fields;

      for (int x = offset; x <= offset+width-1; x++)
        starpu_desc_getaddr( mat.ddescA, y%nb_fields, x );
    }
  }

  /* start timer */
  starpu_mpi_barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    Timer::time_start();
  }
  
  for (int i = 0; i < graphs.size(); i++) {
    const TaskGraph &g = graphs[i];

    for (y = 0; y < g.timesteps; y++) {
      execute_timestep(i, y);
    }
  }

  starpu_task_wait_for_all();
  starpu_mpi_barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    double elapsed = Timer::time_end();
    report_timing(elapsed);
  }
}

void StarPUApp::execute_timestep(size_t idx, long t)
{
  const TaskGraph &g = graphs[idx];
  long offset = g.offset_at_timestep(t);
  long width = g.width_at_timestep(t);
  long dset = g.dependence_set_at_timestep(t);
  matrix_t &mat = mat_array[idx];
  int nb_fields = g.nb_fields;
  
  std::array<starpu_data_handle_t, 10> args;
  payload_t payload;
  
  debug_printf(1, "ts %d, offset %d, width %d, offset+width-1 %d\n", t, offset, width, offset+width-1);
  for (int x = offset; x <= offset+width-1; x++) {
    std::vector<std::pair<long, long> > deps = g.dependencies(dset, x);
    int num_args; 
    starpu_data_handle_t output = starpu_desc_getaddr( mat.ddescA, t%nb_fields, x );
#ifdef ENABLE_PRUNE_MPI_TASK_INSERT
    int has_task = 0;   
    
    if(desc_islocal(mat.ddescA, t%nb_fields, x)) {
      has_task = 1;
    } else if (starpu_mpi_cached_receive(output)) {
      /* We need to invalidate our copy */
      has_task = 1;
    }
    
    if (deps.size() != 0 && t != 0 && has_task != 1) {
      for (std::pair<long, long> dep : deps) {
        for (int i = dep.first; i <= dep.second; i++) {
          if(desc_islocal(mat.ddescA, (t-1)%nb_fields, i)) {
            has_task = 1;
            break;
          }
        }
        if (has_task)
          break;
      }
    }

    debug_printf(1, "rank: %d, has_task: %d, x: %d, t: %d, task_id: %d\n", rank , has_task, x, t, mat.NT * t + x + 1);
    
    if (has_task == 0) {
      continue;
    }

#endif
    
    num_args = 0;
    if (deps.size() == 0) {
      args[num_args++] = output;
      debug_printf(1, "%d[%d] ", x, num_args);
    } else {
      if (t == 0) {
        args[num_args++] = output;
        debug_printf(1, "%d[%d] ", x, num_args);
      } else {
        args[num_args++] = output;
        long last_offset = g.offset_at_timestep(t-1);
        long last_width = g.width_at_timestep(t-1);
        for (std::pair<long, long> dep : deps) {
          for (int i = dep.first; i <= dep.second; i++) {
            if (i >= last_offset && i < last_offset + last_width) {
              args[num_args++] = starpu_desc_getaddr( mat.ddescA, (t-1)%nb_fields, i );
            }
          }
          debug_printf(1, "%d[%d, %d, %d] ", x, num_args, dep.first, dep.second); 
        }
      }
    }
    
    payload.i = t;
    payload.j = x;
    payload.graph = &g;
    payload.graph_id = idx;
    insert_task(num_args, payload, args); 
  }
  debug_printf(1, "\n");
}

void StarPUApp::debug_printf(int verbose_level, const char *format, ...)
{
  if (verbose_level > VERBOSE_LEVEL) {
    return;
  }
  if (rank == 0) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
  }
}


int main(int argc, char **argv)
{
  StarPUApp app(argc, argv);
  app.execute_main_loop();

  return 0;
}
