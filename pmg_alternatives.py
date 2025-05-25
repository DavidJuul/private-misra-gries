"""Alternative versions of differentially private Misra-Gries in practice.

These are older alternative implementations of some of the functions in the
"differentially private Misra-Gries in practice" module (pmg.py).
"""


import math
import random

from sortedcontainers import SortedList


RANDOM = random.SystemRandom()


def misra_gries_unoptimized(stream, sketch_size):
    """Calculate the Misra-Gries sketch of the given stream."""
    sketch = {key: 0 for key in range(-1, -sketch_size - 1, -1)}

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


def misra_gries_with_groups(stream, sketch_size):
    """Calculate the Misra-Gries sketch of the given stream."""
    first_group = SketchGroup(0)
    first_group.elements = SortedList(range(-sketch_size, 0))
    sketch = {key: SketchElement(first_group) for key in range(-sketch_size, 0)}

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


def purely_privatize_misra_gries_unoptimized(sketch, sketch_size, epsilon,
                                             universe_size, element_count,
                                             decrement_count, sensitivity = 2,
                                             offset_counters = True):
    noisy_sketch = {}
    offset = decrement_count - math.floor(element_count / (sketch_size + 1))

    logP = math.log(1 - (1 - math.exp(-epsilon / 2)))
    def geometric():
        return math.floor(math.log(1 - RANDOM.random()) / logP)
    def two_sided_geometric():
        return geometric() - geometric()

    for key in range(universe_size + 1):
        if key in sketch:
            counter = max(sketch[key] + offset, 0)
        else:
            counter = 0
        counter += two_sided_geometric()
        if counter >= 1:
            noisy_sketch[key] = counter

    top_k = sorted(noisy_sketch.items(),
                   key=lambda item: item[1])[-sketch_size:]
    private_sketch = dict(sorted(top_k))

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


def find_threshold_original(epsilon, delta, sensitivity, max_unique_keys = 2):
    return math.ceil(
        1 + 2 * (sensitivity
                 * math.log(2 * (max_unique_keys + 1)
                            * math.exp(epsilon / sensitivity)
                            / ((math.exp(epsilon / sensitivity) + 1) * delta))
                 / epsilon))
