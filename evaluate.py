#!/usr/bin/env python3
"""Evaluation of differentially private Misra-Gries in practice.

This module evaluates everything in the "differentially private Misra-Gries in
practice" module (pmg.py). It performs unit tests where possible, benchmarks of
different iterations of the module, and stochastic testing of the privacy.
"""


import math
import unittest

import pmg


class TestGeometric(unittest.TestCase):
    def test_geometric_varies(self):
        epsilon = 1
        sensitivity = 1
        runs = 100
        geometric = pmg.create_geometric(epsilon, sensitivity)
        noises = {}
        for _ in range(runs):
            noise = geometric()
            if noise in noises:
                noises[noise] += 1
            else:
                noises[noise] = 1
        self.assertGreater(len(noises), 1)

    def test_two_sided_geometric_varies(self):
        epsilon = 1
        sensitivity = 1
        runs = 100
        geometric = pmg.create_two_sided_geometric(epsilon, sensitivity)
        noises = {}
        for _ in range(runs):
            noise = geometric()
            if noise in noises:
                noises[noise] += 1
            else:
                noises[noise] = 1
        self.assertGreater(len(noises), 1)
        self.assertIn(-1, noises)


class TestThreshold(unittest.TestCase):
    def test_threshold_lower_than_union_bound(self):
        epsilon = 0.1
        delta = 1e-6
        union_bound_threshold = math.ceil(
            1 + 2 * (math.log(6 * math.exp(epsilon)
                              / ((math.exp(epsilon) + 1) * delta))
                     / epsilon))
        threshold = pmg.find_threshold(epsilon, delta)
        self.assertGreater(threshold, 1)
        self.assertLess(threshold, union_bound_threshold)


class TestMisraGries(unittest.TestCase):
    def test_sketching(self):
        inputs = [
            ([], 0),
            ([], 1),
            ([1], 1),
            ([1, 2], 1),
            ([1, 2, 3], 1),
            ([4, 3, 3, 2, 1, 2, 3, 4, 4, 4], 4),
            ([4, 3, 3, 2, 1, 2, -1, 3, 4, 4, 4, 5], 3),
        ]
        expected_outputs = [
            ({}, 0, 0),
            ({}, 0, 0),
            ({1: 1}, 1, 0),
            ({1: 0}, 2, 1),
            ({3: 1}, 3, 1),
            ({1: 1, 2: 2, 3: 3, 4: 4}, 10, 0),
            ({2: 0, 3: 1, 4: 2}, 11, 2),
        ]
        for input_, expected_output in zip(inputs, expected_outputs):
            output = pmg.misra_gries(input_[0], input_[1])
            self.assertTupleEqual(output, expected_output)

    def test_max_decrement_count(self):
        unique_elements = 100
        sketch_size = 10
        _, _, decrement_count = pmg.misra_gries(range(1, unique_elements + 1),
                                                sketch_size)
        max_decrements = unique_elements // (sketch_size + 1)
        self.assertEqual(decrement_count, max_decrements)


class TestPrivatizeMisraGries(unittest.TestCase):
    def test_privatized_keys(self):
        sketch = {1: 181, 2: 118, 3: 121, 4: 117, 5: 122}
        sketch_size = len(sketch)
        element_count = 1000
        decrement_count = 100
        epsilon = 10
        delta = element_count / 10
        universe_size = element_count * 10
        user_element_count = 3
        private_sketches = [
            pmg.privatize_misra_gries(sketch, epsilon, delta),
            pmg.purely_privatize_misra_gries(sketch, sketch_size, epsilon,
                                             universe_size, element_count,
                                             decrement_count),
            pmg.privatize_merged(sketch, sketch_size, epsilon, delta),
            pmg.purely_privatize_merged(sketch, sketch_size, epsilon,
                                        universe_size),
            pmg.privatize_user_level(sketch, epsilon, delta,
                                     user_element_count),
            pmg.purely_privatize_user_level(
                sketch, sketch_size, epsilon, universe_size, element_count,
                decrement_count, user_element_count)
        ]
        for private_sketch in private_sketches:
            self.assertSequenceEqual(sketch.keys(), private_sketch.keys())

    def test_approximate_performs_thresholding(self):
        sketch = {1: 170, 2: 120, 3: 1, 4: 2, 5: 210}
        expected_keys = [1, 2, 5]
        epsilon = 1
        delta = 1e-3
        private_sketch = pmg.privatize_misra_gries(sketch, epsilon, delta)
        self.assertListEqual(list(private_sketch.keys()), expected_keys)
        epsilon = 100
        non_private_private_sketch = pmg.privatize_misra_gries(
            sketch, epsilon, delta, threshold=1)
        self.assertSequenceEqual(non_private_private_sketch.keys(),
                                 sketch.keys())

    def test_pure_offsets_counters(self):
        sketch = {1001: 100, 2002: 200, 3003: 300}
        sketch_size = len(sketch)
        element_count = 800
        max_decrements = element_count // (sketch_size + 1)
        decrement_count = max_decrements // 10 * 9
        epsilon = 100
        universe_size = element_count * 10
        private_sketch = pmg.purely_privatize_misra_gries(
            sketch, sketch_size, epsilon, universe_size, element_count,
            decrement_count)
        for key in private_sketch:
            self.assertLess(private_sketch[key], sketch[key])

    def test_pure_upgrades_zeros(self):
        sketch = {10: 4, 20: 7, 30: 15}
        sketch_size = len(sketch)
        element_count = 100
        decrement_count = 0
        epsilon = 1
        universe_size = element_count * 10
        private_sketch = pmg.purely_privatize_misra_gries(
            sketch, sketch_size, epsilon, universe_size, element_count,
            decrement_count)
        for key in private_sketch:
            self.assertNotIn(key, sketch)


class TestMerge(unittest.TestCase):
    def test_merging(self):
        sketch_groups = [
            [{}],
            [{}, {}],
            [{1: 1}],
            [{1: 10, 2: 20}, {3: 30, 4: 40}],
            [{1: 1, 2: 2}, {3: 3, 4: 4, 5: 5}, {6: 6}],
        ]
        sketch_size = 3
        expected_merges = [
            {},
            {},
            {1: 1},
            {2: 10, 3: 20, 4: 30},
            {4: 1, 5: 2, 6: 5},
        ]
        for sketches, expected_merged in zip(sketch_groups, expected_merges):
            merged = pmg.merge(sketches, sketch_size)
            self.assertDictEqual(merged, expected_merged)


def test():
    """Run the unit tests."""
    unittest.main(verbosity=2)


def benchmark():
    """Run the benchmarks."""
    # TODO:


def test_privacy():
    """Run the stochastic testing of the privacy."""
    # TODO:


def main():
    """Run the evaluation of the private Misra-Gries sketching program."""
    test()
    benchmark()
    test_privacy()


if __name__ == "__main__":
    main()
