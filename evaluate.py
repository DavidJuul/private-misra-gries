#!/usr/bin/env python3
"""Evaluation of differentially private Misra-Gries in practice.

This module evaluates everything in the "differentially private Misra-Gries in
practice" module (pmg.py). It performs unit tests where possible, benchmarks of
different iterations of the module, and stochastic testing of the privacy.
"""


import json
import math
import random
import time
import unittest
import unittest.mock

import matplotlib.pyplot as plt

import pmg
import pmg_alternatives


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
            stream = input_[0]
            sketch_size = input_[1]
            output = pmg.misra_gries(stream, sketch_size)
            self.assertTupleEqual(output, expected_output)

    def test_max_decrement_count(self):
        unique_elements = 100
        stream = range(1, unique_elements + 1)
        sketch_size = 10
        _, _, decrement_count = pmg.misra_gries(stream, sketch_size)
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
        universe_size = element_count * 1000
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


def plot_benchmark(title, label, repetitions, function, input_lengths,
                   input_generator):
    print_label = ": {}".format(label) if label else ""
    print("Benchmarking {}{}...".format(title, print_label))
    execution_times = []
    for input_length in input_lengths:
        execution_time = 0
        for _ in range(repetitions):
            input_ = input_generator(input_length)
            start_time = time.perf_counter()
            function(*input_)
            execution_time += time.perf_counter() - start_time
        execution_times.append(execution_time / repetitions)
    plt.plot(input_lengths, execution_times, label=label, marker=".")


def plot_privatization_distribution(title, repetitions, function, sketch,
                                    neighbor_sketch, epsilon, delta,
                                    input_generator):
    print("Testing {}...".format(title))
    plt.clf()
    plt.title(title)
    plt.xlabel("Sketch distribution")
    plt.xticks([])
    plt.ylabel("Count")
    plt.yscale("log")

    # Count the occurring privatized sketches.
    privates = {}
    neighbor_privates = {}
    for _ in range(repetitions):
        private = json.dumps(function(*input_generator(sketch)))
        if private in privates:
            privates[private] += 1
        else:
            privates[private] = 1
        neighbor_private = json.dumps(
            function(*input_generator(neighbor_sketch)))
        if neighbor_private in neighbor_privates:
            neighbor_privates[neighbor_private] += 1
        else:
            neighbor_privates[neighbor_private] = 1

    # Scale the distribution according to the privacy, and count amount of
    # privacy deviations.
    delta_offset = delta * repetitions
    deviations = 0
    for private in privates:
        privates[private] = (math.exp(epsilon) * privates[private]
                             + delta_offset)
    for neighbor_private in neighbor_privates:
        if neighbor_private not in privates:
            privates[neighbor_private] = delta_offset
        if neighbor_privates[neighbor_private] > privates[neighbor_private]:
            deviations += 1

    print("{} had {}/{}={} privacy deviations.".format(
        title, deviations, repetitions, deviations / repetitions))

    # Sort the distribution from most to least occurrences.
    privates = dict(sorted(privates.items(), key=lambda item: item[1],
                           reverse=True))
    neighbor_privates = dict(sorted(neighbor_privates.items(),
                                    key=lambda item: item[1], reverse=True))

    plt.bar(privates.keys(), privates.values(), alpha=0.5,
            label="Scaled distribution of privatized sketch")
    plt.bar(neighbor_privates.keys(), neighbor_privates.values(), alpha=0.5,
            label="Distribution of privatized neighbor sketch")
    plt.legend()


def benchmark_misra_gries_stream_length():
    repetitions = 10
    # stream_lengths = [200 * 2 ** i for i in range(14)]
    stream_lengths = [200 * 2 ** i for i in range(8)]
    sketch_size = 100

    # Benchmark on input with only unique elements.
    def input_generator_without_repeats(stream_length):
        return range(stream_length), sketch_size
    plt.clf()
    title = "Misra-Gries without repeats"
    plt.title(title)
    plt.xlabel("Stream length")
    plt.ylabel("Execution time [s]")
    plot_benchmark(title, "Unoptimized version", repetitions,
                   pmg_alternatives.misra_gries_unoptimized, stream_lengths,
                   input_generator_without_repeats)
    plot_benchmark(title, "Fully-grouped version", repetitions,
                   pmg_alternatives.misra_gries_with_groups, stream_lengths,
                   input_generator_without_repeats)
    plot_benchmark(title, "Zero-group version (final)", repetitions,
                   pmg.misra_gries, stream_lengths,
                   input_generator_without_repeats)
    plt.legend()
    plt.savefig("benchmark_misra_gries_stream_without_repeats.png")

    # Benchmark on input with repating elements.
    def input_generator_with_repeats(stream_length):
        return (map(lambda i: i % sketch_size, range(stream_length)),
                sketch_size)
    plt.clf()
    title = "Misra-Gries with repeats"
    plt.title(title)
    plt.xlabel("Stream length")
    plt.ylabel("Execution time [s]")
    plot_benchmark(title, "Unoptimized version", repetitions,
                   pmg_alternatives.misra_gries_unoptimized, stream_lengths,
                   input_generator_with_repeats)
    plot_benchmark(title, "Fully-grouped version", repetitions,
                   pmg_alternatives.misra_gries_with_groups, stream_lengths,
                   input_generator_with_repeats)
    plot_benchmark(title, "Zero-group version (final)", repetitions,
                   pmg.misra_gries, stream_lengths,
                   input_generator_with_repeats)
    plt.legend()
    plt.savefig("benchmark_misra_gries_stream_with_repeats.png")


def benchmark_misra_gries_sketch_size():
    repetitions = 10
    # sketch_sizes = [10 * 2 ** i for i in range(12)]
    sketch_sizes = [10 * 2 ** i for i in range(8)]
    stream_length = sketch_sizes[-1] * 2

    # Benchmark on input with only unique elements.
    def input_generator_without_repeats(sketch_size):
        return range(stream_length), sketch_size
    plt.clf()
    title = "Misra-Gries without repeats"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Execution time [s]")
    plot_benchmark(title, "Unoptimized version", repetitions,
                   pmg_alternatives.misra_gries_unoptimized, sketch_sizes,
                   input_generator_without_repeats)
    plot_benchmark(title, "Fully-grouped version", repetitions,
                   pmg_alternatives.misra_gries_with_groups, sketch_sizes,
                   input_generator_without_repeats)
    plot_benchmark(title, "Zero-group version (final)", repetitions,
                   pmg.misra_gries, sketch_sizes,
                   input_generator_without_repeats)
    plt.legend()
    plt.savefig("benchmark_misra_gries_sketch_without_repeats.png")

    # Benchmark on input with repating elements.
    def input_generator_with_repeats(sketch_size):
        return (map(lambda i: i % sketch_size, range(stream_length)),
                sketch_size)
    plt.clf()
    title = "Misra-Gries with repeats"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Execution time [s]")
    plot_benchmark(title, "Unoptimized version", repetitions,
                   pmg_alternatives.misra_gries_unoptimized, sketch_sizes,
                   input_generator_with_repeats)
    plot_benchmark(title, "Fully-grouped version", repetitions,
                   pmg_alternatives.misra_gries_with_groups, sketch_sizes,
                   input_generator_with_repeats)
    plot_benchmark(title, "Zero-group version (final)", repetitions,
                   pmg.misra_gries, sketch_sizes,
                   input_generator_with_repeats)
    plt.legend()
    plt.savefig("benchmark_misra_gries_sketch_with_repeats.png")


def benchmark_privatize():
    repetitions = 10
    # sketch_sizes = [10 * 2 ** i for i in range(16)]
    sketch_sizes = [10 * 2 ** i for i in range(12)]
    epsilon = 1
    delta = 1e-6

    def input_generator(sketch_size):
        return {i: i for i in range(sketch_size)}, epsilon, delta
    plt.clf()
    title = "Misra-Gries approximate privatization"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Execution time [s]")
    default_random = pmg.RANDOM
    pmg.random = random.Random()
    plot_benchmark(title, "Pseudo-random sampling", repetitions,
                   pmg.privatize_misra_gries, sketch_sizes, input_generator)
    pmg.random = default_random
    plot_benchmark(title, "Cryptographic sampling (final)", repetitions,
                   pmg.privatize_misra_gries, sketch_sizes, input_generator)
    plt.legend()
    plt.savefig("benchmark_privatize.png")


def benchmark_purely_privatize():
    repetitions = 10
    sketch_size = 100
    sketch = {i: i for i in range(sketch_size)}
    epsilon = 1
    # universe_sizes = [int(sketch_size * 2 ** i) for i in range(2, 14)]
    universe_sizes = [int(sketch_size * 2 ** i) for i in range(2, 8)]
    decrement_count = 0

    def input_generator(universe_size):
        element_count = sketch_size
        return (sketch, sketch_size, epsilon, universe_size, element_count,
                decrement_count)
    plt.clf()
    title = "Misra-Gries pure privatization"
    plt.title(title)
    plt.xlabel("Universe size")
    plt.ylabel("Execution time [s]")
    plot_benchmark(title, "Unoptimized version", repetitions,
                   pmg_alternatives.purely_privatize_misra_gries_unoptimized,
                   universe_sizes, input_generator)
    plot_benchmark(title, "Optimized version (final)", repetitions,
                   pmg.purely_privatize_misra_gries, universe_sizes,
                   input_generator)
    plt.legend()
    plt.savefig("benchmark_purely_privatize.png")


def benchmark_merge():
    repetitions = 10
    # sketch_sizes = [10 * 2 ** i for i in range(16)]
    sketch_sizes = [10 * 2 ** i for i in range(14)]

    def input_generator(sketch_size):
        return ([{i: i for i in range(sketch_size)},
                 {i: i for i in range(sketch_size, sketch_size * 2)}],
                sketch_size)
    plt.clf()
    title = "Misra-Gries merging"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Execution time [s]")
    plot_benchmark(title, "", repetitions, pmg.merge, sketch_sizes,
                   input_generator)
    plt.savefig("benchmark_merge.png")


def benchmark_find_threshold():
    repetitions = 10
    # epsilons = [1 / 2 ** i for i in range(12)]
    epsilons = [1 / 2 ** i for i in range(8)]
    delta = 1e-6

    def input_generator(epsilon):
        return epsilon, delta
    plt.clf()
    title = "Finding thresholds"
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.xscale("log")
    plt.ylabel("Execution time [s]")
    plt.yscale("log")
    plot_benchmark(title, "", repetitions, pmg.find_threshold, epsilons,
                   input_generator)
    plt.savefig("benchmark_find_threshold.png")


def test_privacy_privatize():
    repetitions = 1000
    sketch = {1: 140, 2: 1, 3: 70, 4: 0}
    neighbor_sketch = {1: 140, 3: 70, 5: 0, 6: 0}
    epsilon = 1
    delta = 0.01

    def input_generator(sketch):
        return sketch, epsilon, delta
    title = "Privacy of approximate privatization"
    plot_privatization_distribution(
        title, repetitions, pmg.privatize_misra_gries, sketch, neighbor_sketch,
        epsilon, delta, input_generator)
    plt.savefig("privacy_privatize.png")

    title = "Privacy of approximate privatization with original threshold"
    with unittest.mock.patch("pmg.find_threshold",
                             new=pmg_alternatives.find_threshold_original):
        plot_privatization_distribution(
            title, repetitions, pmg.privatize_misra_gries, sketch,
            neighbor_sketch, epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_original_threshold.png")


def test_privacy_purely_privatize():
    repetitions = 2000
    sketch = {1: 140, 2: 1, 3: 70, 4: 0}
    neighbor_sketch = {1: 140, 3: 70, 5: 0, 6: 0}
    sketch_size = len(sketch)
    epsilon = 1
    delta = 0
    universe_size = 15
    element_count = 221
    decrement_count = 10

    def input_generator(sketch):
        return (sketch, sketch_size, epsilon, universe_size, element_count,
                decrement_count)
    title = "Privacy of pure privatization"
    plot_privatization_distribution(
        title, repetitions, pmg.purely_privatize_misra_gries, sketch,
        neighbor_sketch, epsilon, delta, input_generator)
    plt.savefig("privacy_purely_privatize.png")


def test_privacy_privatize_merged():
    repetitions = 1000
    merged = {1: 140, 2: 1, 3: 70, 4: 0}
    neighbor_merged = {1: 140, 3: 70, 5: 0, 6: 0}
    merged_size = len(merged)
    epsilon = 1
    delta = 0.01

    def input_generator(merged):
        return merged, merged_size, epsilon, delta
    title = "Privacy of approximate privatization of merged"
    plot_privatization_distribution(
        title, repetitions, pmg.privatize_merged, merged, neighbor_merged,
        epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_merged.png")

    title = ("Privacy of approximate privatization of merged with original "
             "threshold")
    with unittest.mock.patch("pmg.find_threshold",
                             new=pmg_alternatives.find_threshold_original):
        plot_privatization_distribution(
            title, repetitions, pmg.privatize_merged, merged, neighbor_merged,
            epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_merged_original_threshold.png")


def test_privacy_purely_privatize_merged():
    repetitions = 1000
    merged = {1: 140, 2: 1, 3: 70, 4: 0}
    neighbor_merged = {1: 140, 3: 70, 5: 0, 6: 0}
    merged_size = len(merged)
    epsilon = 1
    delta = 0
    universe_size = 15

    def input_generator(merged):
        return merged, merged_size, epsilon, universe_size
    title = "Privacy of pure privatization of merged"
    plot_privatization_distribution(
        title, repetitions, pmg.purely_privatize_merged, merged,
        neighbor_merged, epsilon, delta, input_generator)
    plt.savefig("privacy_purely_privatize_merged.png")


def test_privacy_privatize_user_level():
    repetitions = 1000
    sketch = {1: 140, 2: 1, 3: 70, 4: 0}
    neighbor_sketch = {1: 140, 3: 70, 5: 0, 6: 0}
    epsilon = 1
    delta = 0.01
    user_element_count = 5

    def input_generator(sketch):
        return sketch, epsilon, delta, user_element_count
    title = "Privacy of user-level approximate privatization"
    plot_privatization_distribution(
        title, repetitions, pmg.privatize_user_level, sketch, neighbor_sketch,
        epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_user_level.png")


def test_privacy_purely_privatize_user_level():
    repetitions = 1000
    sketch = {1: 140, 2: 1, 3: 70, 4: 0}
    neighbor_sketch = {1: 140, 3: 70, 5: 0, 6: 0}
    sketch_size = len(sketch)
    epsilon = 1
    delta = 0
    universe_size = 15
    element_count = 221
    decrement_count = 10
    user_element_count = 5

    def input_generator(sketch):
        return (sketch, sketch_size, epsilon, universe_size, element_count,
                decrement_count, user_element_count)
    title = "Privacy of user-level pure privatization"
    plot_privatization_distribution(
        title, repetitions, pmg.purely_privatize_user_level, sketch,
        neighbor_sketch, epsilon, delta, input_generator)
    plt.savefig("privacy_purely_privatize_user_level.png")


def test():
    """Run the unit tests."""
    unittest.main(verbosity=2, exit=False)


def benchmark():
    """Run the benchmarks."""
    benchmark_misra_gries_stream_length()
    benchmark_misra_gries_sketch_size()
    benchmark_privatize()
    benchmark_purely_privatize()
    benchmark_merge()
    benchmark_find_threshold()


def test_privacy():
    """Run the stochastic privacy testing."""
    test_privacy_privatize()
    test_privacy_purely_privatize()
    test_privacy_privatize_merged()
    test_privacy_purely_privatize_merged()
    test_privacy_privatize_user_level()
    test_privacy_purely_privatize_user_level()


def main():
    """Run the evaluation of the private Misra-Gries sketching program."""
    print("RUNNING UNIT TESTS...")
    test()
    print("-" * 70)
    print("RUNNING BENCHMARKS...")
    benchmark()
    print("-" * 70)
    print("RUNNING STOCHASTIC PRIVACY TESTING...")
    test_privacy()


if __name__ == "__main__":
    main()
