"""Differentially private Misra-Gries in practice"""


import json
import math
import random
import sys


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


def private_misra_gries(sketch, epsilon, delta):
    private_sketch = {}
    threshold = math.ceil(
        1 + 2 * (math.log(6 * math.exp(epsilon)
                          / ((math.exp(epsilon) + 1) * delta))
                 / epsilon))

    rand = random.SystemRandom()
    logP = math.log(1 - (1 - math.exp(-epsilon)))
    def geometric():
        return math.floor(math.log(1 - rand.random()) / logP)
    def two_sided_geometric():
        return geometric() - geometric()

    eta = two_sided_geometric()
    for key in sorted(sketch):
        counter = sketch[key] + eta + two_sided_geometric()
        if counter >= threshold:
            private_sketch[key] = counter

    return private_sketch


def pure_private_misra_gries(sketch, k, epsilon, element_count,
                             decrement_count, max_key):
    noisy_sketch = {}
    offset = decrement_count - math.floor(element_count / (k + 1))

    rand = random.SystemRandom()
    logP = math.log(1 - (1 - math.exp(-epsilon / 2)))
    def geometric():
        return math.floor(math.log(1 - rand.random()) / logP)
    def two_sided_geometric():
        return geometric() - geometric()

    for key in range(max_key + 1):
        if key in sketch:
            counter = max(sketch[key] + offset, 0)
        else:
            counter = 0
        counter += two_sided_geometric()
        if counter >= 1:
            noisy_sketch[key] = counter

    top_k = sorted(noisy_sketch.items(), key=lambda item: item[1])[-k:]
    private_sketch = dict(sorted(top_k))

    return private_sketch


def merge(sketches, k):
    summed_sketch = sketches[0]
    merged = {}

    for sketch in sketches[1:]:
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

        summed_sketch = merged

    return merged


def private_merge(merged, k, epsilon, delta):
    private_merged = {}
    threshold = math.ceil(
        1 + 2 * (k * math.log(2 * k * math.exp(epsilon / k)
                              / ((math.exp(epsilon / k) + 1) * delta))
                 / epsilon))

    rand = random.SystemRandom()
    logP = math.log(1 - (1 - math.exp(-epsilon / k)))
    def geometric():
        return math.floor(math.log(1 - rand.random()) / logP)
    def two_sided_geometric():
        return geometric() - geometric()

    for key in sorted(merged):
        counter = merged[key] + two_sided_geometric()
        if counter >= threshold:
            private_merged[key] = counter

    return private_merged


def main():
    if len(sys.argv) < 5:
        print("Differentially private Misra-Gries in practice")
        print("Usage:")
        print("  Create a sketch:")
        print("    {} <amount of counters> <epsilon> <delta> <stream file> [output sketch file]"
              .format(sys.argv[0]))
        print("  Merge sketches:")
        print("    {} merge <amount of counters> <epsilon> <delta> <sketch file> [<sketch file> ...]"
              .format(sys.argv[0]))

        return

    def create_sketch():
        k = int(sys.argv[1])
        epsilon = float(sys.argv[2])
        delta = float(sys.argv[3])

        with open(sys.argv[4], encoding="utf8") as stream:
            stream = map(int, stream)
            sketch, element_count, decrement_count = misra_gries(stream, k)

        if delta > 0:
            private_sketch = private_misra_gries(sketch, epsilon, delta)
        else:
            max_key = 100000
            private_sketch = pure_private_misra_gries(
                sketch, k, epsilon, element_count, decrement_count, max_key)

        print("Sketch        :", sketch)
        print("Private sketch:", private_sketch)

        if (len(sys.argv) >= 6):
            with open(sys.argv[5], "w", encoding="utf8") as output:
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
