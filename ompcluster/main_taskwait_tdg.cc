#include <cstdio>
#include <omp.h>

typedef int kmp_int32;
struct ident_t;

extern "C" {
kmp_int32 __kmpc_start_record_task(
    ident_t *, kmp_int32, kmp_int32, kmp_int32);
void __kmpc_end_record_task(
    ident_t *, kmp_int32, kmp_int32, kmp_int32);
}


int main() {
    constexpr int N = 16;
    int values[N];
    int results[N] = {};

    for (int i = 0; i < N; ++i)
        values[i] = i + 10;

    #pragma omp parallel
    #pragma omp single
    {
        int record = __kmpc_start_record_task(nullptr, 0, 0, 0);

        if (record) {
            for (int i = 0; i < N; ++i) {
            #pragma omp target nowait \
                map(to: values[i:1]) \
                map(from: results[i:1]) \
                depend(out: results[i]) \
                firstprivate(i)
            {
                results[i] = values[i];
            }
            }
        }

        __kmpc_end_record_task(nullptr, 0, 0, 0);

        #pragma omp taskwait
    }

    int failures = 0;
    for (int i = 0; i < N; ++i) {
        if (results[i] != i + 10) {
            std::printf("FAIL i=%d expected=%d got=%d\n",
                        i, i + 10, results[i]);
            ++failures;
        }
    }

    if (failures)
        return 1;

    std::printf("MULTI_COPY_TDG_PASS N=%d\n", N);
    return 0;
}
