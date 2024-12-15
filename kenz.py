import numpy as np
import pandas as pd
from complexity import time_and_space_profiler
from tqdm import tqdm

ARRAY_SIZES = [100, 500, 1000, 10000]
TEST_REPEATS = 3


def create_test_data(array_size, order='random'):
    array = np.random.randint(1, 4 * array_size, size=array_size)
    if order == 'ascending':
        return np.sort(array)
    elif order == 'descending':
        return np.sort(array)[::-1]
    return array


def execute_sorting_algorithm(algorithm, data):
    """Helps to run the sorting algorithm and return its results in a consistent format."""
    func_name, (comparisons, moves), time, memory = algorithm(data)
    return (func_name, len(data), data[1], comparisons, moves, time, memory)


@time_and_space_profiler
def sort_with_selection(arr):
    comparisons, moves = 0, 0
    arr = arr.copy()

    for i in range(len(arr)):
        min_index = i
        for j in range(i + 1, len(arr)):
            comparisons += 1
            if arr[j] < arr[min_index]:
                min_index = j
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]
            moves += 1

    return comparisons, moves


@time_and_space_profiler
def sort_with_bubble(arr):
    comparisons, moves = 0, 0
    arr = arr.copy()

    for i in range(len(arr)):
        for j in range(0, len(arr) - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                moves += 1

    return comparisons, moves


@time_and_space_profiler
def sort_with_insertion_exchange(arr):
    comparisons, moves = 0, 0
    arr = arr.copy()

    for i in range(1, len(arr)):
        for j in range(i, 0, -1):
            comparisons += 1
            if arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                moves += 1
            else:
                break

    return comparisons, moves


@time_and_space_profiler
def sort_with_insertion_shift(arr):
    comparisons, moves = 0, 0
    arr = arr.copy()

    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            comparisons += 1
            arr[j + 1] = arr[j]
            j -= 1
            moves += 1
        arr[j + 1] = key

    return comparisons, moves


#  test data
test_data = [
    (size, order, create_test_data(size, order))
    for size in ARRAY_SIZES
    for order in ['random', 'ascending', 'descending']
    for _ in range(TEST_REPEATS)
]

# Run algorithm
results = []
algorithms = [sort_with_selection, sort_with_bubble, sort_with_insertion_exchange, sort_with_insertion_shift]

for size, order, data in tqdm(test_data, desc="Testing Algorithms", leave=True):
    for sort_func in algorithms:
        results.append(execute_sorting_algorithm(sort_func, data))

#   CSV
df = pd.DataFrame(results, columns=['Algorithm', 'Array Size', 'Order', 'Comparisons', 'Moves', 'Time', 'Memory'])
df.to_csv('result.csv', index=False)
