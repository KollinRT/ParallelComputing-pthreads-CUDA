#include <iostream>
#include "arrayFxns.h"


int main(int argc, char* argv[]){
    // Some error handling
    // Ensure exactly three arguments are provided
    if (argc != 3) {
        std::cerr << "Error: Exactly two arguments required." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <M> <N>" << std::endl;
        return 1;
    }

    // Convert arguments to integers with validation
    int M, N;
    std::istringstream issM(argv[1]);
    std::istringstream issN(argv[2]);

    if (!(issM >> M) || !(issN >> N)) {
        std::cerr << "Error: Non-integer input provided." << std::endl;
        return 1;
    }

    // Discard any remaining input beyond the first integer
    char leftover;
    if (issM >> leftover || issN >> leftover) {
        std::cerr << "Error: Extraneous characters after number. Only ints." << std::endl;
        return 1;
    }

    // Ensure M and N do not exceed MAX_SIZE
    M = std::min(M, MAX_SIZE);
    N = std::min(N, MAX_SIZE);

    benchmarkMatrixOps(M, N, M);


    return 0;

}