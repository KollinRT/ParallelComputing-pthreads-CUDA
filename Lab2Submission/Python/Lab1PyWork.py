import argparse
import random
import time

MAX_SIZE = 10000


def generate_random_static_array(M):
    """
    Generate a random static array with size MxM
    :param M:
    :return:
    """
    return [[random.uniform(0.0, 1.0) for _ in range(M)] for _ in range(M)]

def multiply_static_matrix(array1, array2, M):
    """
    Multiply a static array with size MxM
    :param array1:
    :param array2:
    :param M:
    :return:
    """
    result = [[0.0 for _ in range(M)] for _ in range(M)]
    for i in range(M):
        for j in range(M):
            for k in range(M):
                result[i][j] += array1[i][k] * array2[k][j]
    return result

def power_matrix(base, N, M):
    """
    Power a static array with size MxM
    :param base:
    :param N:
    :param M:
    :return:
    """
    result = [[1.0 if i == j else 0.0 for j in range(M)] for i in range(M)]
    temp = [row[:] for row in base]
    while N > 0:
        if N % 2 == 1:
            temp_result = multiply_static_matrix(result, temp, M)
            result = temp_result
        temp_squared = multiply_static_matrix(temp, temp, M)
        temp = temp_squared
        N //= 2
    return result

def main(M, N, trials=5):
    """
    Main function
    :param M: MxM Matrix
    :param N: Nth power to raise
    :param trials: Number of trials for statistical data.
    :return:
    """
    for _ in range(trials):
        matrix = generate_random_static_array(M)
        start_time = time.time()
        result = power_matrix(matrix, N, M)
        elapsed_time = (time.time() - start_time) * 1000  # Time in milliseconds
        print(f"Time for M={M}, N={N}: {elapsed_time:.0f} milliseconds")  # Ensure this is correct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix Operations: Power of a Matrix")
    parser.add_argument('M', type=int, help='Dimension of the square matrix, limited by MAX_SIZE')
    parser.add_argument('N', type=int, help='Exponent to raise the matrix to')
    args = parser.parse_args()
    M = min(args.M, MAX_SIZE)
    N = min(args.N, MAX_SIZE)
    main(M, N)
