#include <cstdio>
#include <omp.h>

int main() {
    constexpr int N = 16;
    int values[N];
    int results[N] = {};

    for (int i = 0; i < N; ++i)
        values[i] = i + 10;

    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < N; ++i) {
            #pragma omp target nowait \
                map(to: values[i:1]) \
                map(from: results[i:1]) \
                firstprivate(i)
            {
                results[i] = values[i];
            }
        }

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

    std::printf("MULTI_COPY_PASS N=%d\n", N);
    return 0;
}
