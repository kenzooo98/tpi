from complexity import time_and_space_profiler
from tqdm import tqdm
import numpy as np
import pandas as pd

# Initialization logic
np.random.seed(42)

# Lengths for testing
lengths = np.arange(1000, 1000000, 20000)
tests_per_length = 5
tests = []

# Generate test cases
for length in lengths:
    for _ in range(tests_per_length):
        val = np.sort(np.random.randint(1, 4 * length, size=length))
        target = np.random.randint(1, 4 * length)
        tests.append((length, val, target))

# Function definitions
@time_and_space_profiler
def linear_search(arr, target_val):
    for i in range(len(arr)):
        if target_val == arr[i]:
            return i + 1
    return len(arr)

@time_and_space_profiler
def optimized_linear_search(arr, target_val):
    comparison = 0
    for i in range(len(arr)):
        comparison += 2
        if target_val == arr[i]:
            comparison -= 1
            break
        elif target_val < arr[i]:
            break
    return comparison

@time_and_space_profiler
def efficient_binary_search(arr, target_val):
    start, end = 0, len(arr) - 1
    comparison = 1
    while start < end:
        mid = (end + start) // 2
        comparison += 2
        if target_val == arr[mid]:
            comparison -= 1
            break
        elif target_val < arr[mid]:
            end = mid - 1
        else:
            start = mid + 1
        comparison += 1
    return comparison

# List of functions to test
search_algorithms = [linear_search, optimized_linear_search, efficient_binary_search]

# Collect results
results = []
for idx, (length, val, target) in tqdm(enumerate(tests), ncols=len(tests)):
    for func in search_algorithms:
        func_name, comparison, time, space = func(val, target)
        results.append((idx, func_name, length, comparison, time, space))

# Create DataFrame and save results
df = pd.DataFrame(results, columns=['id_test', 'function_name', 'array_length', 'comparison', 'time', 'space'])
print(df)
df.to_csv('results.csv', index=False)
