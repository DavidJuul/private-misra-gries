"""Differentially private Misra-Gries in practice"""


import math
import random
import sys


def misra_gries(k, stream):
    sketch = {key: 0 for key in range(-k, 0)}
    zero_group = list(range(-k, 0))
    zero_pointer = 0

    def decrement_all():
        nonlocal zero_group, zero_pointer
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

    return final_sketch


def private_misra_gries(sketch, epsilon, delta):
    private_sketch = {}
    threshold = 1 + 2 * math.ceil(
        math.log(6 * math.exp(epsilon) / ((math.exp(epsilon) + 1) * delta))
        / epsilon)

    rand = random.SystemRandom()
    logP = math.log(1 - (1 - math.exp(-epsilon)))
    def geometric():
        return math.floor(math.log(1 - rand.random()) / logP) + 1
    def two_sided_geometric():
        return geometric() - geometric()

    eta = two_sided_geometric()
    for key in sorted(sketch):
        counter = sketch[key] + eta + two_sided_geometric()
        if counter >= threshold:
            private_sketch[key] = counter

    return private_sketch


def main():
    if len(sys.argv) != 5:
        print("Usage: {} <amount of counters> <epsilon> <delta> <stream file>".format(sys.argv[0]))
        return

    k = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    delta = float(sys.argv[3])
    with open(sys.argv[4], encoding="utf8") as stream:
        stream = map(int, stream)
        sketch = misra_gries(k, stream)

    private_sketch = private_misra_gries(sketch, epsilon, delta)

    print("Sketch        :", sketch)
    print("Private sketch:", private_sketch)


if __name__ == "__main__":
    main()
