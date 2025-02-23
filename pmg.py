"""Differentially private Misra-Gries in practice"""


import math
import sys

from numpy.random import laplace


def misra_gries(k, stream):
    sketch = {key: 0 for key in range(-1, -k - 1, -1)}

    for element in stream:
        if element in sketch:
            sketch[element] += 1
        elif all(map(lambda x: sketch[x] >= 1, sketch)):
            for key in sketch:
                sketch[key] -= 1
        else:
            for key in sketch:
                if sketch[key] == 0:
                    break
            del sketch[key]
            sketch[element] = 1

    for key in sketch:
        if key < 0:
            del sketch[key]

    return sketch


def private_misra_gries(sketch, epsilon, delta):
    private_sketch = {}
    eta = laplace(0, 1 / epsilon)

    for key in sketch:
        counter = sketch[key] + eta + laplace(0, 1 / epsilon)
        if counter >= 1 + 2 * math.log(3 / delta) / epsilon:
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
