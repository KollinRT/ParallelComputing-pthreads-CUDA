//
// Created by kolli on 4/11/2024.
//

#ifndef LAB1_CS530_CPP_WIP_ARRAYFXNS_H
#define LAB1_CS530_CPP_WIP_ARRAYFXNS_H

#include <iostream>
#include <random>
#include <cstddef> // For size_t
#include <cstring> // For memcpy
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>  // Include for std::setprecision and std::fixed



// Static arrays
constexpr int MAX_SIZE = 10000; // Define a maximum size for your array
/// Generate a subsection of the MAX_SIZE by MAX_SIZE array with randomized values in only the first MxM indices
/// \param array the array object
/// \param M the MxM submatrix to work in
void generateRandomStaticArray(double array[MAX_SIZE][MAX_SIZE], int M) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 10.0);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            array[i][j] = distrib(gen);
        }
    }
}

/// Function to multiply the static arrays together in O(n^3)
/// \param array1
/// \param array2
/// \param result the array to hold the results
/// \param M size MxM matrix
void multiplyStaticMatrix(const double** array1, const double** array2, double** result, int M) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i][j] = 0.0; // Ensure the result matrix is initialized to zero before accumulation
            for (int k = 0; k < M; ++k) {
                result[i][j] += array1[i][k] * array2[k][j];
            }
        }
    }
}

/// Function to multiply a submatrix in the larger defined static array
/// \tparam M a template argument to decide if M is M or active_size or MAX_SIZE
/// \param array1 array 1 to be multiplied
/// \param array2 array 2 to be multiplied
/// \param result a static array to hold the results
/// \param active_size the section of the array to work in if not MAX_SIZE x MAX_SIZE
template<std::size_t M>
void multiplyMatrix(const double (&array1)[M][M], const double (&array2)[M][M], double (&result)[M][M], int active_size) {
    for (std::size_t i = 0; i < active_size; ++i) {
        for (std::size_t j = 0; j < active_size; ++j) {
            result[i][j] = 0.0;
            for (std::size_t k = 0; k < active_size; ++k) {
                result[i][j] += array1[i][k] * array2[k][j];
            }
        }
    }
}

/// Function to exponentiate a submatrix in the larger defined static array
/// \tparam M a template argument to decide if M is M or active_size or MAX_SIZE
/// \param base the array to be exponentiated
/// \param N the power to be raised to
/// \param result a static array to hold the results
/// \param active_size the section of the array to work in if not MAX_SIZE x MAX_SIZE
template<std::size_t M>
void powerMatrix(const double (&base)[M][M], int N, double (&result)[M][M], int active_size) {
    for (std::size_t i = 0; i < active_size; ++i) {
        for (std::size_t j = 0; j < active_size; ++j) {
            result[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    double temp[M][M];
    std::memcpy(temp, base, sizeof(temp));

    while (N > 0) {
        if (N % 2 == 1) {
            double tempResult[M][M];
            multiplyMatrix<M>(result, temp, tempResult, active_size);
            std::memcpy(result, tempResult, sizeof(tempResult));
        }
        double tempSquared[M][M];
        multiplyMatrix<M>(temp, temp, tempSquared, active_size);
        std::memcpy(temp, tempSquared, sizeof(tempSquared));
        N /= 2;
    }
}

// Some dynamic memory code
/// Generate a dynamic random array
/// \param array pointer to the array of pointers for dynamic allocation
/// \param size size of the array.
void generateRandomArray(double** array, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            array[i][j] = distrib(gen);
        }
    }
}

/// Function to delete the dynamically allocated array to prevent memory leaks
/// \param matrix dynamically allocated pointers
/// \param M size MxM matrix
void freeDynamicMatrix(double** matrix, int M) {
    for (int i = 0; i < M; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

/// Function to allocate MxM dynamically allocated array
/// \param M size MxM array
/// \return the array object
double** allocateDynamicMatrix(int M) {
    double** matrix = new double*[M];
    for (int i = 0; i < M; ++i) {
        matrix[i] = new double[M];
    }
    return matrix;
}

/// Function to multiply two dynamicallly allocated matrices
/// \param matrix1 matrix 1
/// \param matrix2 matrix 2
/// \param result matrix to hold the results
/// \param M size MxM array
void multiplyDynamicMatrix(double** matrix1, double** matrix2, double** result, int M) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i][j] = 0.0;
            for (int k = 0; k < M; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

/// Function to copy a dynamically allocated array
/// \param src array to be copied
/// \param dest destination of array copying
/// \param M size MxM array
void copyDynamicMatrix(double** src, double** dest, int M) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

/// Function to compute the power of a dynamically allocated matrix
/// \param base matrix to be exponentiated
/// \param N exponent of raising
/// \param result location for results of matrix raised
/// \param M size MxM array
void powerDynamicMatrix(double** base, int N, double** result, int M) {
    // Initialize result as the identity matrix
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    double** temp = new double*[M];
    for (int i = 0; i < M; i++) {
        temp[i] = new double[M];
        for (int j = 0; j < M; j++) {
            temp[i][j] = base[i][j];
        }
    }

    while (N > 0) {
        if (N % 2 == 1) {
            double** tempResult = new double*[M];
            for (int i = 0; i < M; i++) {
                tempResult[i] = new double[M];
            }
            multiplyDynamicMatrix(result, temp, tempResult, M);
            copyDynamicMatrix(tempResult, result, M);
            // Free tempResult after use
            for (int i = 0; i < M; i++) {
                delete[] tempResult[i];
            }
            delete[] tempResult;
        }

        // Square the matrix 'temp' and store the result back in 'temp'
        double** tempSquared = new double*[M];
        for (int i = 0; i < M; i++) {
            tempSquared[i] = new double[M];
        }
        multiplyDynamicMatrix(temp, temp, tempSquared, M);
        copyDynamicMatrix(tempSquared, temp, M);
        // Free tempSquared after use
        for (int i = 0; i < M; i++) {
            delete[] tempSquared[i];
        }
        delete[] tempSquared;

        N /= 2;
    }

    // Free temp
    for (int i = 0; i < M; i++) {
        delete[] temp[i];
    }
    delete[] temp;
}

/// Function to initialize a dynamic array
/// \param M size of MxM array
/// \return a pointer to a pointer of the dynamic array that is MxM randomized between 0 and 10.
double** initializeDynamicArray(int M) {
    // Allocate memory for the 2D array
    double **A = (double **) malloc(M * sizeof(double *));
    for (int i = 0; i < M; ++i) {
        A[i] = (double *) malloc(M * sizeof(double));
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            A[i][j] = distrib(gen);
        }
    }

    return A;
}




// Define the node structure for a 2D doubly-linked list
typedef struct Node {
    double data;
    struct Node *right;
    struct Node *left;
    struct Node *up;
    struct Node *down;
} Node;

// Function to create a new Node
Node* createNode(double data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (!newNode) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    newNode->data = data;
    newNode->right = NULL;
    newNode->left = NULL;
    newNode->up = NULL;
    newNode->down = NULL;
    return newNode;
}

/// Function to set data value at a given node.
/// \param matrix the node doubly-linked list pointer
/// \param i row index
/// \param j column index
/// \param data data value to be set
void setData(Node* matrix, int i, int j, double data) {
    Node* tempRow = matrix;
    for (int x = 0; x < i; ++x) {
        tempRow = tempRow->down;
        if (!tempRow) return; // Out of bounds
    }

    Node* tempCol = tempRow;
    for (int y = 0; y < j; ++y) {
        tempCol = tempCol->right;
        if (!tempCol) return; // Out of bounds
    }

    if (tempCol) {
        tempCol->data = data;
    }
}

/// Function to insert a node at the end of a row
/// \param row
/// \param data
void insertAtRowEnd(Node** row, double data) {
    Node* newNode = createNode(data);
    if (*row == NULL) {
        *row = newNode;
        return;
    }
    Node* temp = *row;
    while (temp->right != NULL) {
        temp = temp->right;
    }
    temp->right = newNode;
    newNode->left = temp;
}

/// Function to create an MxM matrix
/// \param M size of MxM array
/// \return
Node* createMatrix(int M) {
    if (M <= 0) return NULL;

    Node* matrix = NULL; // This will point to the top-left node
    Node* aboveRow = NULL; // This will keep track of the nodes in the row above the current one

    // Creating rows
    for (int i = 0; i < M; ++i) {
        Node* currentRow = NULL; // Pointer to the first node in the current row
        Node* lastNode = NULL; // Pointer to the last node created in the current row, for linking left and right

        // Creating columns
        for (int j = 0; j < M; ++j) {
            Node* newNode = createNode(i * M + j + 1);
            if (!matrix) {
                matrix = newNode; // The very first node
            }

            if (!currentRow) {
                currentRow = newNode; // The first node in the current row
            } else {
                lastNode->right = newNode; // Linking the last node created to the new one
                newNode->left = lastNode; // Linking the new node back to the last one
            }

            if (aboveRow) {
                // Linking the new node to the one above it
                Node* temp = aboveRow;
                for (int k = 0; k < j; ++k) {
                    temp = temp->right; // Move to the column we're currently filling
                }
                temp->down = newNode;
                newNode->up = temp;
            }

            lastNode = newNode; // Update the lastNode pointer for the next iteration
        }

        // Move aboveRow to the row we just finished creating
        aboveRow = currentRow;
    }

    return matrix;
}

void printMatrix(Node* matrix, int M) {
    Node* rowPtr = matrix;
    for (int i = 0; i < M; ++i) {
        Node* colPtr = rowPtr;
        for (int j = 0; j < M; ++j) {
            std::cout << std::fixed << std::setprecision(2) << colPtr->data << " ";
            colPtr = colPtr->right;
        }
        std::cout << std::endl;
        rowPtr = rowPtr->down;
    }
}


/// How to get the data at a node (i,j) in the doubly-linked list representation
/// \param matrix
/// \param i
/// \param j
/// \return data value at node
int getData(Node* matrix, int i, int j) {
    Node* currentRow = matrix;
    // Move down to the i-th row
    for (int row = 0; row < i; ++row) {
        if (currentRow == NULL) {
            printf("Row out of bounds.\n");
            return -1; // Indicate an error or out-of-bounds access
        }
        currentRow = currentRow->down;
    }
    // Move right to the j-th column
    Node* currentCol = currentRow;
    for (int col = 0; col < j; ++col) {
        if (currentCol == NULL) {
            printf("Column out of bounds.\n");
            return -1; // Indicate an error or out-of-bounds access
        }
        currentCol = currentCol->right;
    }
    return currentCol ? currentCol->data : -1;
}

/// Generate a randomized array dynamically
/// \param array a double pointer to the dynamic initialized array
/// \param M size of MxM array
void generateDynamicRandomArray(double** array, int M) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 10.0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            array[i][j] = distrib(gen);
        }
    }
}


/// Create a doubly-linked list matrix for a size MxM matrix of doubles for trials
/// \param matrix address of the Node struct initialized already.
/// \param M size of MxM matrix
void fillMatrixWithRandomDoubles(Node* matrix, int M) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 10.0);

    Node* rowPtr = matrix;
    for (int i = 0; i < M; ++i) {
        Node* colPtr = rowPtr;
        for (int j = 0; j < M; ++j) {
            colPtr->data = distrib(gen); // Assign random double to node
            colPtr = colPtr->right;
        }
        rowPtr = rowPtr->down;
    }
}

/// A function to multiply two MxM matrix doubly-linked lists
/// \param matrix1
/// \param matrix2
/// \param result
/// \param M the size of MxM matrix.
void multiplyMatrixNodes(Node* matrix1, Node* matrix2, Node* result, int M) {
//    std::cout << "Multiplying Matrices:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0.0; // Initialize sum for this element
            for (int k = 0; k < M; ++k) {
                double a = getData(matrix1, i, k);
                double b = getData(matrix2, k, j);
                sum += a * b; // Accumulate product into sum
//                std::cout << "Multiplying " << a << " (matrix1[" << i << "][" << k << "]) * "
//                          << b << " (matrix2[" << k << "][" << j << "]) = " << sum << "\n";
            }
            setData(result, i, j, sum); // Set accumulated sum in result matrix
//            std::cout << "Setting result[" << i << "][" << j << "] = " << sum << "\n";
        }
    }
//    std::cout << "Resultant Matrix after multiplication:\n";
//    printMatrix(result, M);
}




/// Function to copy matrix values from source to destination
/// \param source the original matrix
/// \param destination the name of the matrix to be copied to
/// \param M the size of the MxM array.
void copyMatrix(Node* source, Node* destination, int M) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double value = getData(source, i, j); // Assuming implementation of getData
            setData(destination, i, j, value); // Assuming implementation of setData
        }
    }
}

/// Function to free all nodes in a matrix
/// \param matrix the matrix to be freed
/// \param M the size of the MxM matrix
void freeMatrix(Node* matrix, int M) {
    Node* row = matrix;
    for (int i = 0; i < M; i++) {
        Node* col = row;
        Node* nextRow = row->down;
        for (int j = 0; j < M; j++) {
            Node* nextCol = col->right;
            free(col);
            col = nextCol;
        }
        row = nextRow;
    }
}

/// Function to raise the power of a matrix.
/// \param base Matrix to be power raise
/// \param N power exponent
/// \param result the matrix that the resulting power raised array will be stored in.
/// \param M The size of the MxM matrix.
void powerMatrixNode(Node* base, int N, Node* result, int M) {
    // Initialize result as the identity matrix
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            setData(result, i, j, (i == j) ? 1.0 : 0.0);
        }
    }

    Node* temp = createMatrix(M);
    copyMatrix(base, temp, M);  // Assuming this function correctly copies matrix data

    while (N > 0) {
        if (N % 2 == 1) {
            Node* tempResult = createMatrix(M);
            multiplyMatrixNodes(result, temp, tempResult, M);
            copyMatrix(tempResult, result, M);  // Assuming this copies the data correctly
            freeMatrix(tempResult, M);  // Ensure memory is freed
        }
        Node* tempSquared = createMatrix(M);
        multiplyMatrixNodes(temp, temp, tempSquared, M);
        copyMatrix(tempSquared, temp, M);
        freeMatrix(tempSquared, M);

        N /= 2;
    }
    freeMatrix(temp, M);
}

// Code for benchmarking series now
void benchmarkMatrixOps(int M, int N, int active_size) {
    // Prepare output file
    std::ofstream outputFile("matrix_benchmark_results.csv", std::ios_base::app);

    // Variables for timing
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    long long duration;
    std::vector<long long> durations(3, 0); // Use a vector to store cumulative durations for the three tests


    //// 1. Static Arrays Benchmark
    //double myStaticArray[MAX_SIZE][MAX_SIZE];
    //double myStaticArrayResult[MAX_SIZE][MAX_SIZE];

    //for (int i = 0; i < 100; ++i) {
    //    generateRandomStaticArray(myStaticArray, M);
    //    auto start = std::chrono::high_resolution_clock::now();
    //    powerMatrix<MAX_SIZE>(myStaticArray, N, myStaticArrayResult, active_size);
    //    auto end = std::chrono::high_resolution_clock::now();
    //    durations[0] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //}
    //long long avgDurationStatic = durations[0] / 100;
    //outputFile << "Static Array," << M << "," << N << ",Power," << avgDurationStatic << "\n";


    // 2. Dynamic Arrays Benchmark
    //double** myDynamicArray = allocateDynamicMatrix(M);
    //double** myDynamicArrayResult = allocateDynamicMatrix(M);
    //for (int i = 0; i < 100; ++i) {
    //    auto start = std::chrono::high_resolution_clock::now();
    //    powerDynamicMatrix(myDynamicArray, N, myDynamicArrayResult, M);
    //    auto end = std::chrono::high_resolution_clock::now();
    //    durations[1] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //}
    //long long avgDurationDynamic = durations[1] / 100;
    //outputFile << "Dynamic Array," << M << "," << N << ",Power," << avgDurationDynamic << "\n";
    //freeDynamicMatrix(myDynamicArray, M);
    //freeDynamicMatrix(myDynamicArrayResult, M);

    // 2. Dynamic Arrays Benchmark
    double** myDynamicArray = allocateDynamicMatrix(M);
    double** myDynamicArrayResult = allocateDynamicMatrix(M);
    std::vector<long long> durations2(5, 0);  // Pre-fill with zeros

    for (int i = 0; i < 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        powerDynamicMatrix(myDynamicArray, N, myDynamicArrayResult, M);
        auto end = std::chrono::high_resolution_clock::now();
        durations2[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        outputFile << "Dynamic Array," << M << "," << N << ",Power," << durations2[i] << "\n"; // Write each timing to the output file
    }

    freeDynamicMatrix(myDynamicArray, M);
    freeDynamicMatrix(myDynamicArrayResult, M);






    //// 3. Doubly-Linked Lists Benchmark
    //Node* nodeMatrix = createMatrix(M);
    //Node* nodeResults = createMatrix(M);
    //fillMatrixWithRandomDoubles(nodeMatrix, M);

    //for (int i = 0; i < 100; ++i) {
    //    auto start = std::chrono::high_resolution_clock::now();
    //    powerMatrixNode(nodeMatrix, N, nodeResults, M);
    //    auto end = std::chrono::high_resolution_clock::now();
    //    durations[2] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //}
    //long long avgDurationDoublyLinked = durations[2] / 100;
    //outputFile << "Doubly-Linked List," << M << "," << N << ",Power," << avgDurationDoublyLinked << "\n";


    //freeMatrix(nodeMatrix, M);
    //freeMatrix(nodeResults, M);

    outputFile.close();
    std::cout << "Benchmarking completed. Results saved to matrix_benchmark_results.csv\n";
}

#endif //LAB1_CS530_CPP_WIP_ARRAYFXNS_H
