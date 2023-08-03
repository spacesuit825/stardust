import numpy as np
import os

def load_data(filepath: set) -> np.array:
    # Intermediate storage for the coordinates
    out = []

    # load file
    with open(filepath) as f:
        lines = f.readlines()

        for line in lines:
            # Remove whitespace (\n, etc.)
            stripped_line = line.strip()

            # Split the line into tokens
            tokenised_line = stripped_line.split(' ')

            # Skip any comments
            if tokenised_line[0] == "#":
                continue

            # Assume all other lines have the shape [str, ' ', str]
            x_value = float(tokenised_line[0]) # <-- Use float() to convert from the str to float type

            out.append(x_value)

    # Convert to numpy type
    return np.array(out)

def compute_average(out):
    return np.mean(out)

files_to_compare = ["blocksweep.txt", "serialweep.txt"]

base_path = "C:/Users/lachl/OneDrive/Documents/c++/stardust/profiling/"

compare = []
for name in files_to_compare:
    path = base_path + name

    out = load_data(path)
    avg = compute_average(out)

    compare.append(avg)

ranking, files_sorted = zip(*sorted(zip(compare, files_to_compare)))

print("------------Profiling---------------")
for n, place in enumerate(ranking):
    print(f"{n}: {files_sorted[n]}, Avg. Time: {place}")
