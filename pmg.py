"""Differentially private Misra-Gries in practice"""


import math
import secrets
import sys

from diffprivlib.mechanisms import Geometric
from sortedcontainers import SortedList


def misra_gries(k, stream):
    first_group = SketchGroup(0)
    first_group.elements = SortedList(range(-k, 0))
    sketch = {key: SketchElement(first_group) for key in range(-k, 0)}

    for key in stream:
        if key in sketch:
            element = sketch[key]
            if len(element.group.elements) == 1 and (not element.group.next
                                                     or element.group.next.count_diff > 1):
                element.group.count_diff += 1
                if element.group.next:
                    element.group.next.count_diff -= 1
            else:
                element.group.elements.remove(key)
                if len(element.group.elements) == 0:
                    if element.group.prev:
                        element.group.prev.next = element.group.next
                    if element.group.next:
                        element.group.next.prev = element.group.prev
                if element.group.next and element.group.next.count_diff == 1:
                    if len(element.group.elements) == 0:
                        element.group.next.count_diff += element.group.count_diff
                    element.group = element.group.next
                else:
                    new_group = SketchGroup(1)
                    if len(element.group.elements) == 0:
                        new_group.prev = element.group.prev
                    else:
                        new_group.prev = element.group
                    if element.group.next:
                        element.group.next.count_diff -= 1
                        new_group.next = element.group.next
                        element.group.next.prev = new_group
                    element.group.next = new_group
                    element.group = new_group
                if len(first_group.elements) == 0:
                    first_group = first_group.next
                element.group.elements.add(key)
        elif first_group.count_diff >= 1:
            first_group.count_diff -= 1
        else:
            min_zero_key = first_group.elements.pop(0)
            del sketch[min_zero_key]
            element = SketchElement()
            sketch[key] = element
            if first_group.next and first_group.next.count_diff == 1:
                element.group = first_group.next
            else:
                new_group = SketchGroup(1)
                new_group.prev = first_group
                if first_group.next:
                    new_group.next = first_group.next
                    first_group.next.prev = new_group
                first_group.next = new_group
                element.group = new_group
            element.group.elements.add(key)
            if len(first_group.elements) == 0:
                first_group = first_group.next

    group = first_group
    count = 0
    while group:
        count += group.count_diff
        for key in group.elements:
            sketch[key] = count
        group = group.next

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


class SketchElement:
    def __init__(self, group=None):
        self.group = group


class SketchGroup:
    def __init__(self, count_diff):
        self.count_diff = count_diff
        self.elements = SortedList()
        self.prev = None
        self.next = None


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
