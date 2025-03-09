"""Differentially private Misra-Gries in practice"""


import math
import secrets
import sys

from diffprivlib.mechanisms import Geometric
from sortedcontainers import SortedList


def misra_gries(k, stream):
    zero_group = SortedList(range(-k, 0))
    sketch = {key: 0 for key in range(-k, 0)}

    for element in stream:
        if element in sketch:
            if sketch[element] == 0:
                zero_group.remove(element)
            sketch[element] += 1
        elif len(zero_group) == 0:
            for key in sketch:
                sketch[key] -= 1
                if sketch[key] == 0:
                    zero_group.add(key)
        else:
            removed_key = zero_group.pop(0)
            del sketch[removed_key]
            sketch[element] = 1

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
    geometric = Geometric(epsilon=epsilon,
                          random_state=secrets.SystemRandom())
    eta = geometric.randomise(0)

    for key in sorted(sketch):
        counter = sketch[key] + eta + geometric.randomise(0)
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
