#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h> // For gettimeofday function

#include <cuda_runtime.h>

typedef struct {
    int index;
    int size;
    float **matA;
    float **matB;
    float **result;
    pthread_barrier_t *barrier;
} ThreadData;

pthread_mutex_t **mutexes;

/// Function to multiply based on based on threads in columns of product matrix
/// \param arg
/// \return
void *mult_row(void* arg) {
    ThreadData *data = (ThreadData *)arg;
    int n = data->size;
    int row = data->index;

    if (row >= n) {
        printf("Row index %d out of bounds for size %d\n", row, n);
        return NULL;
    }

    for (int col = 0; col < n; col++) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += data->matA[row][k] * data->matB[k][col];
        }
        data->result[row][col] = sum;
    }
    pthread_barrier_wait(data->barrier);
    return NULL;
}

/// Function to multiply matrixs based on threads in rows of product matrix
/// \param arg
/// \return
void *mult_column(void* arg) {
    ThreadData *data = (ThreadData *)arg;
    int n = data->size;
    int col = data->index;

    if (data->matA == NULL || data->matB == NULL || data->result == NULL) {
        printf("Invalid matrix pointers in mult_column\n");
        return NULL;
    }

    for (int row = 0; row < n; row++) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += data->matA[row][k] * data->matB[k][col];
        }
        pthread_mutex_lock(&mutexes[row][col]);
        data->result[row][col] = sum;
        pthread_mutex_unlock(&mutexes[row][col]);
    }
    pthread_barrier_wait(data->barrier);
    return NULL;
}

///  Function to multiply based on based on threads for every element of product matrix
/// \param arg
/// \return
void *mult_element(void* arg) {
    ThreadData *data = (ThreadData *)arg;
    int n = data->size;
    int row = data->index / n;
    int col = data->index % n;
    float sum = 0.0;

    for (int k = 0; k < n; k++) {
        sum += data->matA[row][k] * data->matB[k][col];
    }

    pthread_mutex_lock(&mutexes[row][col]);
    data->result[row][col] = sum;
    pthread_mutex_unlock(&mutexes[row][col]);

    pthread_barrier_wait(data->barrier);
    return NULL;
}

/// Initialize the mutexes to help prevent race conditions
/// \param n
void initializeMutexes(int n) {
    mutexes = (pthread_mutex_t **)malloc(n * sizeof(pthread_mutex_t *));
    for (int i = 0; i < n; i++) {
        mutexes[i] = (pthread_mutex_t *)malloc(n * sizeof(pthread_mutex_t));
        for (int j = 0; j < n; j++) {
            pthread_mutex_init(&mutexes[i][j], NULL);
        }
    }
}

/// Remove the mutexes
/// \param n
void freeMutexes(int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            pthread_mutex_destroy(&mutexes[i][j]);
        }
        free(mutexes[i]);
    }
    free(mutexes);
}

/// Dynamically allocate a matrix and initialize it to (0,1]
/// \param n
/// \return
float** allocateMatrix(int n) {
    float **matrix = (float **)malloc(n * sizeof(float *));
    if (!matrix) return NULL;
    for (int i = 0; i < n; i++) {
        matrix[i] = (float *)malloc(n * sizeof(float));
        if (!matrix[i]) {
            for (int j = 0; j < i; j++) free(matrix[j]); // Free previously allocated rows
            free(matrix);
            return NULL;
        }
        for (int j = 0; j < n; j++) {
//            matrix[i][j] = 1.0; // Initialize each element to 1.0
            matrix[i][j] = (float)(rand() + 1) / (RAND_MAX ); // TODO: I think this is it?
        }
    }
    return matrix;
}

/// Free the matrices
/// \param matrix double pointer pointing to the pointer of pointers for the dynamically allocated arrays
/// \param n size of array nxn
void freeMatrix(float **matrix, int n) {
    if (matrix != NULL) {
        for (int i = 0; i < n; i++) {
            free(matrix[i]);
        }
        free(matrix);
    }
}

/// Function used for debugging to print the matrix
/// \param matrix double pointer pointing to the pointer of pointers for the dynamically allocated arrays
/// \param n size of array nxn
void printMatrix(float **matrix, int n) {
    printf("Matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/// Function to swap the addresses of two matrices to swap values.
/// \param mat1
/// \param mat2
void swapMatrices(float ***mat1, float ***mat2) {
    float **temp = *mat1;
    *mat1 = *mat2;
    *mat2 = temp;
}

/// Function to call for multithreaded matrix multiplication
/// \param A Array to utilize
/// \param n size nxn matrix
/// \param exponent How many times to raise
/// \param mult_function function pointer to matrix multiplication method
void matrixPower(float **A, int n, int exponent, void *(*mult_function)(void *)) {
    int threadCount = (mult_function == mult_element) ? n * n : n;
    pthread_t *threads = (pthread_t *)malloc(threadCount * sizeof(pthread_t));
    ThreadData *threadData = (ThreadData *)malloc(threadCount * sizeof(ThreadData));
    pthread_barrier_t barrier;

    if (!threads || !threadData) {
        fprintf(stderr, "Failed to allocate memory for thread data or threads.\n");
        exit(EXIT_FAILURE);
    }

    pthread_barrier_init(&barrier, NULL, threadCount);

    float **result = allocateMatrix(n);
    float **temp = allocateMatrix(n);

    // Initialize result to identity matrix for matrix power calculations
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = (i == j) ? 1.0 : 0.0;
            temp[i][j] = A[i][j];
        }
    }

    while (exponent > 0) {
        if (exponent % 2 == 1) {
            float **tempResult = allocateMatrix(n);
            if (!tempResult) {
                fprintf(stderr, "Failed to allocate temporary result matrix.\n");
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < threadCount; i++) {
                threadData[i] = (ThreadData){i, n, temp, result, tempResult, &barrier};
                pthread_create(&threads[i], NULL, mult_function, &threadData[i]);
            }

            for (int i = 0; i < threadCount; i++) {
                pthread_join(threads[i], NULL);
            }

            swapMatrices(&result, &tempResult);
            freeMatrix(tempResult, n);
        }

        if (exponent > 1) {
            float **newTemp = allocateMatrix(n);
            if (!newTemp) {
                fprintf(stderr, "Failed to allocate new temp matrix.\n");
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < threadCount; i++) {
                threadData[i] = (ThreadData){i, n, temp, temp, newTemp, &barrier};
                pthread_create(&threads[i], NULL, mult_function, &threadData[i]);
            }

            for (int i = 0; i < threadCount; i++) {
                pthread_join(threads[i], NULL);
            }

            freeMatrix(temp, n);
            temp = newTemp;
        }

        exponent >>= 1;
    }

//    printMatrix(result, n);

    freeMatrix(temp, n);
    freeMatrix(result, n);
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(threadData);
}

/// Function to benchmark times for pthread multithreaded multiplication of arrays.
/// \param maxThreads number of threads to utilize
/// \param power power to raise to
/// \param numRuns number of times to run the matrix raising.
void benchmarkMatrixPower(int maxThreads, int power, int numRuns) {
    struct timeval start, end;
    float **A = allocateMatrix(maxThreads);
    initializeMutexes(maxThreads);

//    printf("Matrix A (Initial):\n");
//    printMatrix(A, maxThreads);

    FILE *file = fopen("matrix_benchmark_results.csv", "a");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file for writing.\n");
        return;
    }

    //fprintf(file, "Method,Matrix Size,Power,Time (microseconds)\n");

//    // Benchmarking mult_element
//    for (int i = 0; i < numRuns; i++) {
//    gettimeofday(&start, NULL);
//    matrixPower(A, maxThreads, power, mult_element);
//    gettimeofday(&end, NULL);
//    long elapsedTime = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000;
//    fprintf(file, "Element,%d,%d,%ld\n", maxThreads, power, elapsedTime);
//    }

    // Benchmarking mult_row
    for (int i = 0; i < numRuns; i++) {
        gettimeofday(&start, NULL);
        matrixPower(A, maxThreads, power, mult_row);
        gettimeofday(&end, NULL);
        long elapsedTime = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000;
        fprintf(file, "Element,%d,%d,%ld\n", maxThreads, power, elapsedTime);
    }

    // Benchmarking mult_column
    for (int i = 0; i < numRuns; i++) {
        gettimeofday(&start, NULL);
        matrixPower(A, maxThreads, power, mult_column);
        gettimeofday(&end, NULL);
        long elapsedTime = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000;
        fprintf(file, "Element,%d,%d,%ld\n", maxThreads, power, elapsedTime);
    }
    fclose(file);

    // Clean up
    freeMutexes(maxThreads);
    freeMatrix(A, maxThreads);
}

// CUDA
/// Function to perform row wise multiplication with CUDA
/// \param A
/// \param B
/// \param C
/// \param numARows
/// \param numAColumns
/// \param numBColumns
__global__ void row_wise_mult(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numARows) {
        for (int col = 0; col < numBColumns; col++) {
            float sum = 0.0;
            for (int k = 0; k < numAColumns; k++) {
                sum += A[row * numAColumns + k] * B[k * numBColumns + col];
            }
            C[row * numBColumns + col] = sum;
        }
    }
}

/// Function to perform column wise multiplication with CUDA
/// \param A
/// \param B
/// \param C
/// \param numARows
/// \param numAColumns
/// \param numBColumns
__global__ void column_wise_mult(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < numBColumns) {
        for (int row = 0; row < numARows; row++) {
            float sum = 0.0;
            for (int k = 0; k < numAColumns; k++) {
                sum += A[row * numAColumns + k] * B[k * numBColumns + col];
            }
            C[row * numBColumns + col] = sum;
        }
    }
}

/// Function to perform element wise multiplication with CUDA
/// \param A Pointer to matrix A
/// \param B Pointer to matrix B
/// \param C Pointer to results matrix C
/// \param numARows
/// \param numAColumns
/// \param numBColumns
__global__ void element_wise_mult(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numARows && col < numBColumns) {
        float sum = 0.0;
        for (int k = 0; k < numAColumns; k++) {
            sum += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}


void handleCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))

/// Function to print the matrix size for how CUDA arrays are handled here; FOR DEBUGGING
/// \param matrix pointer to matrix
/// \param size size of MxM matrix
void printMatrixCUDA(float *matrix, int size) {
    printf("Matrix (%d x %d):\n", size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%7.2f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void initializeMatricesRandomly(float *A, int elements) {
    srand(time(NULL)); // Seed the random number generator with current time
    for (int i = 0; i < elements; i++) {
        A[i] = (float)(rand() + 1) / (RAND_MAX ); // Random float between 0 (exclusive) and 1 (inclusive), I think? # TODO: FIX?
    }
}


enum MultMethod {
    ROW_WISE,
    COLUMN_WISE,
    ELEMENT_WISE
};


/// Function to perform benchmarking for the CUDA power raise of a size x size matrix, for numRuns
/// \param size to create size x size Matrix
/// \param power power to be raised for
/// \param method method to be used for arrays
/// \param numRuns number of times ran
void benchmarkMatrixPowerCUDA(int size, int power, MultMethod method, int numRuns) {
    int numARows = size, numAColumns = size, numBColumns = size;
    float *A, *C, *D;
    float *d_A, *d_C, *d_D;
    size_t sizeA = numARows * numAColumns * sizeof(float);
    size_t sizeC = numARows * numBColumns * sizeof(float);

    A = (float *)malloc(sizeA);
    C = (float *)malloc(sizeC);
    D = (float *)malloc(sizeC);

    // Initialize A with random values between 0 and 1
    initializeMatricesRandomly(A, numARows * numAColumns);

    HANDLE_ERROR(cudaMalloc((void **)&d_A, sizeA));
    HANDLE_ERROR(cudaMalloc((void **)&d_C, sizeC));
    HANDLE_ERROR(cudaMalloc((void **)&d_D, sizeC));

    HANDLE_ERROR(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_C, d_A, sizeC, cudaMemcpyDeviceToDevice)); // Initialize d_C with A

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numBColumns + threadsPerBlock.x - 1) / threadsPerBlock.x, (numARows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    FILE *file = fopen("matrix_benchmark_results.csv", "a");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing.\n");
        return;
    }

    for (int run = 0; run < numRuns; run++) {
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start));

        for (int i = 1; i < power; i++) {
            switch (method) {
                case ROW_WISE:
                    row_wise_mult<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_D, numARows, numAColumns, numBColumns);
                    break;
                case COLUMN_WISE:
                    column_wise_mult<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_D, numARows, numAColumns, numBColumns);
                    break;
                case ELEMENT_WISE:
                    element_wise_mult<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_D, numARows, numAColumns, numBColumns);
                    break;
            }
            HANDLE_ERROR(cudaPeekAtLastError());
            HANDLE_ERROR(cudaDeviceSynchronize());

            float *temp = d_C;
            d_C = d_D;
            d_D = temp;
        }

        HANDLE_ERROR(cudaEventRecord(stop));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float milliseconds = 0;
        HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Record the result
        fprintf(file, "%s,%d,%d,%f\n",
                (method == ROW_WISE) ? "CUDA-Row-wise" :
                (method == COLUMN_WISE) ? "CUDA-Column-wise" : "CUDA-Element-wise",
                size, power, milliseconds);
    }

    fclose(file);

    HANDLE_ERROR(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));
    // printMatrixCUDA(C, size); // Optionally print matrix C

    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_D);
    free(A);
    free(C);
    free(D);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <MAX_THREADS> <Power>\n", argv[0]);
        return 1;
    }

    int maxThreads = atoi(argv[1]);
    int power = atoi(argv[2]);

    if (maxThreads <= 0 || power <= 0) {
        fprintf(stderr, "Both arguments must be positive integers.\n");
        return 1;
    }

    int size = maxThreads;


    /* Decide which to run */
    benchmarkMatrixPower(maxThreads, power, 5);
    benchmarkMatrixPowerCUDA(size, power, ROW_WISE, 5);
    benchmarkMatrixPowerCUDA(size, power, COLUMN_WISE, 5);
    benchmarkMatrixPowerCUDA(size, power, ELEMENT_WISE, 5);

    return 0;
}
