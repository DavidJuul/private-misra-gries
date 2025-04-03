"""Differentially private Misra-Gries in practice"""


import json
import math
import random
import sys

from numpy.random import binomial


def misra_gries(stream, k):
    sketch = {key: 0 for key in range(-k, 0)}
    zero_group = list(range(-k, 0))
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

    def insert_element(element):
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
        sketch[element] = 1

    for element in stream:
        element_count += 1
        if element in sketch:
            sketch[element] += 1
        elif zero_pointer == len(zero_group):
            decrement_all()
        else:
            insert_element(element)

    final_sketch = {}
    for key in sorted(sketch):
        if key >= 0:
            final_sketch[key] = sketch[key]

    return final_sketch, element_count, decrement_count


def private_misra_gries(sketch, epsilon, delta, sensitivity=1, threshold=-1,
                        global_noise=True):
    private_sketch = {}
    if threshold == -1:
        threshold = math.ceil(
            1 + 2 * (math.log(6 * math.exp(epsilon)
                              / ((math.exp(epsilon) + 1) * delta))
                     / epsilon))

    rand = random.SystemRandom()
    logP = math.log(1 - (1 - math.exp(-epsilon / sensitivity)))
    def geometric():
        return math.floor(math.log(1 - rand.random()) / logP)
    def two_sided_geometric():
        return geometric() - geometric()

    eta = two_sided_geometric() if global_noise else 0
    for key in sorted(sketch):
        counter = sketch[key] + eta + two_sided_geometric()
        if counter >= threshold:
            private_sketch[key] = counter

    return private_sketch


def pure_private_misra_gries(sketch, k, epsilon, element_count,
                             decrement_count, universe_size):
    noisy_sketch = {}
    offset = decrement_count - math.floor(element_count / (k + 1))
    threshold = math.ceil(
        math.log((1 + math.exp(-epsilon / 2)) * k / universe_size)
        / math.log(math.exp(-epsilon / 2)))

    rand = random.SystemRandom()
    logP = math.log(1 - (1 - math.exp(-epsilon / 2)))
    def geometric():
        return math.floor(math.log(1 - rand.random()) / logP)
    def two_sided_geometric():
        return geometric() - geometric()

    for key in sketch:
        counter = sketch[key] + offset + two_sided_geometric()
        if counter >= threshold:
            noisy_sketch[key] = counter

    upgrade_count = binomial(universe_size, k / universe_size)
    while upgrade_count > 0:
        key = rand.randrange(0, universe_size)
        if key not in noisy_sketch:
            noisy_sketch[key] = threshold + geometric()
            upgrade_count -= 1

    top_k = sorted(noisy_sketch.items(), key=lambda item: item[1])[-k:]
    private_sketch = dict(sorted(top_k))

    return private_sketch


def merge(sketches, k):
    merged = sketches[0]

    for sketch in sketches[1:]:
        summed_sketch = merged
        for key in sketch:
            if key in summed_sketch:
                summed_sketch[key] += sketch[key]
            else:
                summed_sketch[key] = sketch[key]

        if len(summed_sketch) > k:
            offset = sorted(summed_sketch.items(),
                            key=lambda item: item[1])[-(k + 1)][1]
        else:
            offset = 0
        for key in summed_sketch:
            if summed_sketch[key] > offset:
                merged[key] = summed_sketch[key] - offset

    return merged


def private_merge(merged, k, epsilon, delta):
    threshold = math.ceil(
        1 + 2 * (k * math.log(2 * k * math.exp(epsilon / k)
                              / ((math.exp(epsilon / k) + 1) * delta))
                 / epsilon))
    return private_misra_gries(merged, epsilon, delta, k, threshold, False)


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

    def create_sketch():
        k = int(sys.argv[1])
        epsilon = float(sys.argv[2])
        delta = float(sys.argv[3])
        if delta > 0:
            file = sys.argv[4]
        else:
            universe_size = float(sys.argv[4])
            file = sys.argv[5]

        with open(file, encoding="utf8") as stream:
            stream = map(int, stream)
            sketch, element_count, decrement_count = misra_gries(stream, k)

        if delta > 0:
            private_sketch = private_misra_gries(sketch, epsilon, delta)
        else:
            private_sketch = pure_private_misra_gries(
                sketch, k, epsilon, element_count, decrement_count,
                universe_size)

        print("Sketch        :", sketch)
        print("Private sketch:", private_sketch)

        if ((delta > 0 and len(sys.argv) >= 6) or len(sys.argv) >= 7):
            output_file = sys.argv[5] if delta > 0 else sys.argv[6]
            with open(output_file, "w", encoding="utf8") as output:
                json.dump(sketch, output)

    def merge_sketches():
        k = int(sys.argv[2])
        epsilon = float(sys.argv[3])
        delta = float(sys.argv[4])
        sketch_files = sys.argv[5:]

        sketches = []
        for file in sketch_files:
            with open(file, encoding="utf8") as input_:
                sketches.append({int(key): counter for key, counter
                                 in json.load(input_).items()})

        merged = merge(sketches, k)

        private_merged = private_merge(merged, k, epsilon, delta)

        print("Merged        :", merged)
        print("Private merged:", private_merged)

    if sys.argv[1] == "merge":
        merge_sketches()
    else:
        create_sketch()


if __name__ == "__main__":
    main()
