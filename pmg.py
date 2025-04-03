"""Differentially private Misra-Gries in practice"""


import json
import math
import random
import sys

from numpy.random import binomial


RANDOM = random.SystemRandom()


def misra_gries(stream, sketch_size):
    sketch = {key: 0 for key in range(-sketch_size, 0)}
    zero_group = list(range(-sketch_size, 0))
    zero_pointer = 0
    element_count = 0
    decrement_count = 0

    def decrement_all():
        nonlocal zero_group, zero_pointer, decrement_count
        decrement_count += 1
        zero_group = []
        for key in sketch:
            sketch[key] -= 1
            if sketch[key] == 0:
                zero_group.append(key)
        zero_group.sort()
        zero_pointer = 0

    def insert_key(key):
        nonlocal zero_pointer
        while True:
            removed_key = zero_group[zero_pointer]
            zero_pointer += 1
            if sketch[removed_key] == 0:
                break
            if zero_pointer == len(zero_group):
                decrement_all()
                return
        del sketch[removed_key]
        sketch[key] = 1

    for element in stream:
        element_count += 1
        if element in sketch:
            sketch[element] += 1
        elif zero_pointer == len(zero_group):
            decrement_all()
        else:
            insert_key(element)

    final_sketch = {}
    for key in sorted(sketch):
        if key >= 0:
            final_sketch[key] = sketch[key]

    return final_sketch, element_count, decrement_count


def privatize_misra_gries(sketch, epsilon, delta, sensitivity=1, threshold=-1,
                          add_global_noise=True):
    if threshold == -1:
        threshold = math.ceil(
            1 + 2 * (math.log(6 * math.exp(epsilon)
                              / ((math.exp(epsilon) + 1) * delta))
                     / epsilon))
    two_sided_geometric = create_two_sided_geometric(epsilon, sensitivity)
    global_noise = two_sided_geometric() if add_global_noise else 0

    private_sketch = {}
    for key in sorted(sketch):
        counter = sketch[key] + global_noise + two_sided_geometric()
        if counter >= threshold:
            private_sketch[key] = counter

    return private_sketch


def purely_privatize_misra_gries(sketch, sketch_size, epsilon, universe_size,
                                 element_count, decrement_count, sensitivity=2,
                                 offset_counts=True):
    offset = (decrement_count - math.floor(element_count / (sketch_size + 1))
              if offset_counts else 0)
    threshold = math.ceil(
        math.log((1 + math.exp(-epsilon / sensitivity)) * sketch_size
                 / universe_size)
        / math.log(math.exp(-epsilon / sensitivity)))
    geometric = create_geometric(epsilon, sensitivity)
    two_sided_geometric = create_two_sided_geometric(epsilon, sensitivity)

    noisy_sketch = {}
    for key in sketch:
        counter = sketch[key] + offset + two_sided_geometric()
        if counter >= threshold:
            noisy_sketch[key] = counter

    upgrade_count = binomial(universe_size, sketch_size / universe_size)
    while upgrade_count > 0:
        key = RANDOM.randrange(0, universe_size)
        if key not in noisy_sketch:
            noisy_sketch[key] = threshold + geometric()
            upgrade_count -= 1

    top_k = sorted(noisy_sketch.items(),
                   key=lambda item: item[1])[-sketch_size:]
    private_sketch = dict(sorted(top_k))

    return private_sketch


def merge(sketches, sketch_size):
    merged = sketches[0]
    for sketch in sketches[1:]:
        summed_sketch = merged
        for key in sketch:
            if key in summed_sketch:
                summed_sketch[key] += sketch[key]
            else:
                summed_sketch[key] = sketch[key]

        if len(summed_sketch) > sketch_size:
            offset = sorted(summed_sketch.items(),
                            key=lambda item: item[1])[-(sketch_size + 1)][1]
            for key in summed_sketch:
                if summed_sketch[key] > offset:
                    merged[key] = summed_sketch[key] - offset
        else:
            merged = dict(summed_sketch)

    return merged


def privatize_merged(merged, sketch_size, epsilon, delta):
    threshold = math.ceil(
        1 + 2 * (sketch_size * math.log(2 * sketch_size
                                        * math.exp(epsilon / sketch_size)
                              / ((math.exp(epsilon / sketch_size) + 1)
                                 * delta))
                 / epsilon))
    return privatize_misra_gries(merged, epsilon, delta, sketch_size,
                                 threshold, False)


def purely_privatize_merged(merged, sketch_size, epsilon, universe_size):
    return purely_privatize_misra_gries(merged, sketch_size, epsilon,
                                        universe_size, None, None, sketch_size,
                                        False)


def create_geometric(epsilon, sensitivity):
    log_alpha = math.log(math.exp(-epsilon / sensitivity))
    return lambda: math.floor(math.log(1 - RANDOM.random()) / log_alpha)


def create_two_sided_geometric(epsilon, sensitivity):
    geometric = create_geometric(epsilon, sensitivity)
    return lambda: geometric() - geometric()


def create_sketch():
    sketch_size = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    delta = float(sys.argv[3])
    if delta > 0:
        file = sys.argv[4]
    else:
        universe_size = int(sys.argv[4])
        file = sys.argv[5]

    # Create the sketch from the file stream.
    with open(file, encoding="utf8") as stream:
        stream = map(int, stream)
        sketch, element_count, decrement_count = misra_gries(stream,
                                                             sketch_size)

    # Privatize the sketch.
    if delta > 0:
        private_sketch = privatize_misra_gries(sketch, epsilon, delta)
    else:
        private_sketch = purely_privatize_misra_gries(
            sketch, sketch_size, epsilon, element_count, decrement_count,
            universe_size)

    # Output the sketch.
    print("Sketch        :", sketch)
    print("Private sketch:", private_sketch)
    if ((delta > 0 and len(sys.argv) >= 6) or len(sys.argv) >= 7):
        output_file = sys.argv[5] if delta > 0 else sys.argv[6]
        with open(output_file, "w", encoding="utf8") as output:
            json.dump(sketch, output)


def merge_sketches():
    sketch_size = int(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    if delta > 0:
        sketch_files = sys.argv[5:]
    else:
        universe_size = int(sys.argv[5])
        sketch_files = sys.argv[6:]

    # Merge the sketches from the files.
    sketches = []
    for file in sketch_files:
        with open(file, encoding="utf8") as input_:
            sketches.append({int(key): counter for key, counter
                             in json.load(input_).items()})
    merged = merge(sketches, sketch_size)

    # Privatize the merged sketch.
    if delta > 0:
        private_merged = privatize_merged(merged, sketch_size, epsilon, delta)
    else:
        private_merged = purely_privatize_merged(merged, sketch_size, epsilon,
                                                 universe_size)

    # Output the merged sketch.
    print("Merged        :", merged)
    print("Private merged:", private_merged)


def main():
    if len(sys.argv) < 5:
        print("Differentially private Misra-Gries in practice")
        print("Usage:")
        print("  Create an (epsilon, delta)-private sketch:")
        print(f"    {sys.argv[0]} <sketch size> <epsilon> <delta> "
              "<stream file> [output sketch file]")
        print("  Create an (epsilon, 0)-private sketch:")
        print(f"    {sys.argv[0]} <sketch size> <epsilon> 0 <universe size> "
              "<stream file> [output sketch file]")
        print("  Merge sketches with (epsilon, delta)-privacy:")
        print(f"    {sys.argv[0]} merge <sketch size> <epsilon> <delta> "
              "<sketch file> [<sketch file> ...]")
        print("  Merge sketches with (epsilon, 0)-privacy:")
        print(f"    {sys.argv[0]} merge <sketch size> <epsilon> 0 "
              "<universe size> <sketch file> [<sketch file> ...]")
        return

    if sys.argv[1] == "merge":
        merge_sketches()
    else:
        create_sketch()


if __name__ == "__main__":
    main()
