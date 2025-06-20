#!/usr/bin/env python3
"""Evaluation of Differentially Private Misra-Gries in Practice.

This module evaluates everything in the "Differentially Private Misra-Gries in
Practice" module (pmg.py). It performs unit tests where possible, benchmarks of
different iterations of the module, and stochastic testing of the privacy.
"""


import json
import math
import random
import tempfile
import time
import unittest
import unittest.mock

import matplotlib.pyplot as plt
import scipy.stats

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
            unoptimized_output = pmg_alternatives.misra_gries_unoptimized(
                stream, sketch_size)
            self.assertTupleEqual(unoptimized_output, expected_output)
            with_groups_output = pmg_alternatives.misra_gries_with_groups(
                stream, sketch_size)
            self.assertTupleEqual(with_groups_output, expected_output)

    def test_max_decrement_count(self):
        unique_elements = 100
        stream = range(1, unique_elements + 1)
        sketch_size = 10
        max_decrements = unique_elements // (sketch_size + 1)
        _, _, decrement_count = pmg.misra_gries(stream, sketch_size)
        self.assertEqual(decrement_count, max_decrements)
        _, _, unoptimized_decrement_count = (
            pmg_alternatives.misra_gries_unoptimized(stream, sketch_size))
        self.assertEqual(unoptimized_decrement_count, max_decrements)
        _, _, with_groups_decrement_count = (
            pmg_alternatives.misra_gries_with_groups(stream, sketch_size))
        self.assertEqual(with_groups_decrement_count, max_decrements)


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
            pmg_alternatives.purely_privatize_misra_gries_unoptimized(
                sketch, sketch_size, epsilon, universe_size, element_count,
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
            sketch_files = []
            for sketch in sketches:
                sketch_file = tempfile.NamedTemporaryFile().name
                with open(sketch_file, "w", encoding="utf8") as file:
                    json.dump(sketch, file)
                sketch_files.append(sketch_file)
            merged_quickselect = pmg.merge(sketch_files, sketch_size)
            self.assertDictEqual(merged_quickselect, expected_merged)
            with unittest.mock.patch(
                "pmg.find_kth_largest",
                new=pmg_alternatives.find_kth_largest_quickselect):
                merged_quickselect = pmg.merge(sketch_files, sketch_size)
                self.assertDictEqual(merged_quickselect, expected_merged)


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
                                    input_generator,
                                    purely_privatization_offset = 0):
    e_epsilon = math.exp(epsilon)
    count_sums = isinstance(function(*input_generator(sketch)), dict)
    unique_counter = list(sketch.keys())[-2]
    count_unique_counter = (unique_counter not in neighbor_sketch
                            and sketch[unique_counter] == 1)

    print("Testing {}...".format(title))
    plt.clf()
    plt.title(title)
    plt.xlabel("Sketch distribution")
    plt.xticks([])
    plt.ylabel("Count")
    plt.yscale("log")

    # Count the occurring privatized sketches, the counter sums, the accurate
    # releases of the sketch's first counter, and the releases of the unique
    # counter.
    privates = {}
    neighbor_privates = {}
    counter_sum = 0
    neighbor_counter_sum = 0
    original_first = list(next(iter(sketch.items())))
    original_first[1] += purely_privatization_offset
    original_first_releases = 0
    neighbor_original_first_releases = 0
    unique_counter_releases = 0
    neighbor_unique_counter_releases = 0
    for _ in range(repetitions):
        private = function(*input_generator(sketch))
        if count_sums:
            counter_sum += sum(private.values())
        if (original_first[0] in private
            and (not isinstance(private, dict)
                 or private[original_first[0]] == original_first[1])):
            original_first_releases += 1
        if count_unique_counter and unique_counter in private:
            unique_counter_releases += 1
        private = json.dumps(private)
        if private in privates:
            privates[private] += 1
        else:
            privates[private] = 1

        neighbor_private = function(*input_generator(neighbor_sketch))
        if count_sums:
            neighbor_counter_sum += sum(neighbor_private.values())
        if (original_first[0] in neighbor_private
            and (not isinstance(neighbor_private, dict)
                 or neighbor_private[original_first[0]] == original_first[1])):
            neighbor_original_first_releases += 1
        if count_unique_counter and unique_counter in neighbor_private:
            neighbor_unique_counter_releases += 1
        neighbor_private = json.dumps(neighbor_private)
        if neighbor_private in neighbor_privates:
            neighbor_privates[neighbor_private] += 1
        else:
            neighbor_privates[neighbor_private] = 1

    # Scale the distribution according to the privacy, and count amount of
    # privacy deviations.
    deviations = 0
    wilson_violations = 0
    for neighbor_private in neighbor_privates:
        if neighbor_private not in privates:
            privates[neighbor_private] = 0
        lower_bound = wilson_interval(privates[neighbor_private],
                                        repetitions)[0]
        upper_bound = wilson_interval(neighbor_privates[neighbor_private],
                                        repetitions)[1]
        if lower_bound / upper_bound > e_epsilon:
            wilson_violations += 1
    for private in privates:
        privates[private] = e_epsilon * privates[private]
    for neighbor_private in neighbor_privates:
        if neighbor_privates[neighbor_private] > privates[neighbor_private]:
            deviations += math.ceil(neighbor_privates[neighbor_private]
                                    - privates[neighbor_private])

    print("{} had {}/{}={} privacy deviations.".format(
        title, deviations, repetitions, deviations / repetitions))
    print("{} had {} Wilson-based privacy violations.".format(
        title, wilson_violations))
    if count_sums:
        total_sum_ratio = max(counter_sum, 1) / max(neighbor_counter_sum, 1)
        print("{} had total sum ratio {}/{}{}e^epsilon.".format(
            title, counter_sum, neighbor_counter_sum,
            "<" if total_sum_ratio <= e_epsilon
            and 1 / total_sum_ratio <= e_epsilon else ">"
        ))
    original_first_release_ratio = (max(original_first_releases, 1)
                                    / max(neighbor_original_first_releases, 1))
    print("{} had {}/{}{}e^epsilon accurate releases of the first counter."
          "".format(
              title, original_first_releases, neighbor_original_first_releases,
              "<" if original_first_release_ratio <= e_epsilon
              and 1 / original_first_release_ratio <= e_epsilon else ">"
    ))
    unique_counter_release_ratio = (max(unique_counter_releases, 1)
                                    / max(neighbor_unique_counter_releases, 1))
    if count_unique_counter:
        print("{} had {}/{}{}e^epsilon releases of the unique counter."
            "".format(
                title, unique_counter_releases,
                neighbor_unique_counter_releases,
                "<" if unique_counter_release_ratio <= e_epsilon
                and 1 / unique_counter_release_ratio <= e_epsilon else ">"
        ))

    # Sort the distribution from most to least occurrences.
    privates = dict(sorted(privates.items(), key=lambda item: item[1],
                           reverse=True))
    neighbor_privates = dict(sorted(neighbor_privates.items(),
                                    key=lambda item: item[1], reverse=True))

    plt.bar(privates.keys(), privates.values(), alpha=0.5,
            label="Scaled distribution of privatized sketch")
    plt.bar(neighbor_privates.keys(), neighbor_privates.values(), alpha=0.5,
            label="Distribution of privatized neighbor sketch")
    plt.legend(loc="upper right")


def plot_accuracy(title, label, repetitions, function, input_lengths,
                 input_generator, inaccuracy_generator = None, sci_y = True):
    print_label = ": {}".format(label) if label else ""
    print("Testing {}{}...".format(title, print_label))
    inaccuracies = []
    if inaccuracy_generator:
        beta = 0.05
        accuracy_deviations = 0
        max_deviations = 0
        total_inaccuracy = 0
        total_max_inaccuracy = 0
    for input_length in input_lengths:
        input_ = input_generator(input_length)
        sketch = input_[0]
        if inaccuracy_generator:
            min_inaccuracy, max_inaccuracy = inaccuracy_generator(input_length,
                                                                  beta = beta)
        inaccuracy = 0
        for _ in range(repetitions):
            private_sketch = function(*input_)
            for key in sketch:
                if key not in private_sketch:
                    private_sketch[key] = 0
            if inaccuracy_generator:
                max_deviations += len(private_sketch)
                total_max_inaccuracy += max_inaccuracy * len(private_sketch)
            for key in private_sketch:
                if key not in sketch:
                    sketch[key] = 0
                error = private_sketch[key] - sketch[key]
                inaccuracy += abs(error)
                if inaccuracy_generator:
                    if error < min_inaccuracy:
                        accuracy_deviations += 1
                    if error > max_inaccuracy:
                        accuracy_deviations += 1
        inaccuracy /= repetitions
        inaccuracies.append(inaccuracy)
        if inaccuracy_generator:
            total_inaccuracy += inaccuracy
    if inaccuracy_generator:
        deviation_ratio = accuracy_deviations / max_deviations
        print("{}{} had {}/{}={}{}beta accuracy deviations.".format(
            title, print_label, accuracy_deviations, max_deviations,
            deviation_ratio, "<" if deviation_ratio < beta else ">"))
        print("{}{} had total inaccuracy {}{}{} max expected inaccuracy."
              "".format(
                  title, print_label, total_inaccuracy,
                  "<" if total_inaccuracy <= total_max_inaccuracy else ">",
                  total_max_inaccuracy))

    plt.plot(input_lengths, inaccuracies, label=label, marker=".")
    if sci_y:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def wilson_interval(occurrences, experiments, alpha = 0.01):
    proportion = occurrences / experiments
    critical = scipy.stats.norm.ppf(1 - alpha / 2)
    denominator = 1 + critical ** 2 / experiments
    center = proportion + critical ** 2 / (2 * experiments)
    margin = critical * math.sqrt(proportion * (1 - proportion) / experiments
                                  + critical ** 2 / (4 * experiments ** 2))
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return lower, upper


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
    plt.legend(loc="upper right")
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
    plt.legend(loc="upper right")
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
    plt.legend(loc="upper right")
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
    plt.legend(loc="upper right")
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
    with unittest.mock.patch("pmg.random", new=random.Random()):
        plot_benchmark(title, "Pseudo-random sampling", repetitions,
                       pmg.privatize_misra_gries, sketch_sizes,
                       input_generator)
    plot_benchmark(title, "Cryptographic sampling (final)", repetitions,
                   pmg.privatize_misra_gries, sketch_sizes, input_generator)
    plt.legend(loc="upper right")
    plt.savefig("benchmark_privatize.png")


def benchmark_purely_privatize():
    repetitions = 10
    sketch_sizes = [10 * 2 ** i for i in range(10)]
    # sketch_sizes = [10 * 2 ** i for i in range(6)]
    epsilon = 1
    universe_size = sketch_sizes[-1] * 10
    decrement_count = 0

    def input_generator(sketch_size):
        return ({i: i for i in range(sketch_size)}, sketch_size, epsilon,
                universe_size, sketch_size, decrement_count)
    plt.clf()
    title = "Misra-Gries pure privatization (sketch size)"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Execution time [s]")
    plot_benchmark(title, "Unoptimized version", repetitions,
                   pmg_alternatives.purely_privatize_misra_gries_unoptimized,
                   sketch_sizes, input_generator)
    plot_benchmark(title, "Optimized version (final)", repetitions,
                   pmg.purely_privatize_misra_gries, sketch_sizes,
                   input_generator)
    plt.legend(loc="upper right")
    plt.savefig("benchmark_purely_privatize_sketch.png")

    sketch_size = 100
    sketch = {i: i for i in range(sketch_size)}
    # universe_sizes = [int(sketch_size * 2 ** i) for i in range(2, 14)]
    universe_sizes = [int(sketch_size * 2 ** i) for i in range(2, 8)]
    def input_generator2(universe_size):
        element_count = sketch_size
        return (sketch, sketch_size, epsilon, universe_size, element_count,
                decrement_count)
    plt.clf()
    title = "Misra-Gries pure privatization (universe size)"
    plt.title(title)
    plt.xlabel("Universe size")
    plt.ylabel("Execution time [s]")
    plot_benchmark(title, "Unoptimized version", repetitions,
                   pmg_alternatives.purely_privatize_misra_gries_unoptimized,
                   universe_sizes, input_generator2)
    plot_benchmark(title, "Optimized version (final)", repetitions,
                   pmg.purely_privatize_misra_gries, universe_sizes,
                   input_generator2)
    plt.legend(loc="upper right")
    plt.savefig("benchmark_purely_privatize_universe.png")


def benchmark_merge():
    repetitions = 10
    # sketch_sizes = [10 * 2 ** i for i in range(14)]
    sketch_sizes = [10 * 2 ** i for i in range(10)]

    sketch_file1 = tempfile.NamedTemporaryFile()
    sketch_file2 = tempfile.NamedTemporaryFile()
    def input_generator(sketch_size):
        with open(sketch_file1.name, "w", encoding="utf8") as file:
            json.dump({i: i for i in range(sketch_size)}, file)
        with open(sketch_file2.name, "w", encoding="utf8") as file:
            json.dump({i: i for i in range(sketch_size, sketch_size * 2)},
                      file)
        return [sketch_file1.name, sketch_file2.name], sketch_size
    plt.clf()
    title = "Misra-Gries merging"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Execution time [s]")

    with unittest.mock.patch(
        "pmg.find_kth_largest",
        new=pmg_alternatives.find_kth_largest_quickselect):
        plot_benchmark(title, "Quickselect", repetitions, pmg.merge,
                       sketch_sizes, input_generator)
    plot_benchmark(title, "Sorting (final version)", repetitions, pmg.merge,
                   sketch_sizes, input_generator)
    plt.legend(loc="upper right")
    plt.savefig("benchmark_merge.png")


def benchmark_find_threshold():
    repetitions = 10
    # epsilons = [1 / 2 ** i for i in range(12)]
    epsilons = [1 / 2 ** i for i in range(8)]
    delta = 1e-6

    def input_generator(epsilon):
        return epsilon, delta
    plt.clf()
    title = "Finding thresholds (epsilon)"
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.xscale("log")
    plt.ylabel("Execution time [s]")
    plt.yscale("log")
    plot_benchmark(title, "", repetitions, pmg.find_threshold, epsilons,
                   input_generator)
    plt.savefig("benchmark_find_threshold_epsilon.png")

    epsilon = 1
    deltas = [1 / 100 ** i for i in range(24)]
    def input_generator2(delta):
        return epsilon, delta
    plt.clf()
    title = "Finding thresholds (delta)"
    plt.title(title)
    plt.xlabel("Delta")
    plt.xscale("log")
    plt.ylabel("Execution time [s]")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plot_benchmark(title, "", repetitions, pmg.find_threshold, deltas,
                   input_generator2)
    plt.savefig("benchmark_find_threshold_delta.png")


def test_privacy_privatize():
    # repetitions = 10000
    repetitions = 1000
    sketch = {0: 140, 1: 70, 2: 1, 3: 0}
    neighbor_sketch = {0: 140, 1: 70, 4: 0, 5: 0}
    epsilon = 1
    delta = 1 / (sum(sketch.values()) - 1 + 9) / 10

    def input_generator(sketch):
        return sketch, epsilon, delta
    title = "Privacy of approximate privatization"
    plot_privatization_distribution(
        title, repetitions, pmg.privatize_misra_gries, sketch, neighbor_sketch,
        epsilon, delta, input_generator)
    plt.savefig("privacy_privatize.png")

    title = "Privacy of approximate privatization with loose threshold"
    with unittest.mock.patch("pmg.find_threshold",
                             new=pmg_alternatives.find_threshold_original):
        plot_privatization_distribution(
            title, repetitions, pmg.privatize_misra_gries, sketch,
            neighbor_sketch, epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_original_threshold.png")

    title = "Privacy of approximate privatization by keys"
    plot_privatization_distribution(
        title, repetitions,
        lambda *args: list(pmg.privatize_misra_gries(*args)), sketch,
        neighbor_sketch, epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_keys.png")

    sketch = {0: 99, 1: 49}
    neighbor_sketch = {0: 100, 1: 50}
    title = "Privacy of approximate privatization lowered counters"
    plot_privatization_distribution(
        title, repetitions, pmg.privatize_misra_gries, sketch, neighbor_sketch,
        epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_lowered.png")

    title = "Privacy of approximate privatization lowered counters by keys"
    plot_privatization_distribution(
        title, repetitions,
        lambda *args: list(pmg.privatize_misra_gries(*args)), sketch,
        neighbor_sketch, epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_lowered_keys.png")


def test_privacy_purely_privatize():
    # repetitions = 20000
    repetitions = 2000
    sketch = {0: 40, 1: 1, 2: 0}
    neighbor_sketch = {0: 40, 3: 0, 4: 0}
    sketch_size = len(sketch)
    epsilon = 2
    delta = 0
    universe_size = 12
    element_count = 9 + sketch[0]
    decrement_count = 2
    privatization_offset = (decrement_count
                            - math.floor(element_count / (sketch_size + 1)))

    def input_generator(sketch):
        _element_count = (element_count - 1 if sketch == neighbor_sketch
                          else element_count)
        return (sketch, sketch_size, epsilon, universe_size, _element_count,
                decrement_count)
    title = "Privacy of pure privatization"
    plot_privatization_distribution(
        title, repetitions, pmg.purely_privatize_misra_gries, sketch,
        neighbor_sketch, epsilon, delta, input_generator, privatization_offset)
    plt.savefig("privacy_purely_privatize.png")

    title = "Privacy of pure privatization by keys"
    plot_privatization_distribution(
        title, repetitions,
        lambda *args: list(pmg.purely_privatize_misra_gries(*args)), sketch,
        neighbor_sketch, epsilon, delta, input_generator, privatization_offset)
    plt.savefig("privacy_purely_privatize_keys.png")

    sketch = {0: 39, 1: 29}
    neighbor_sketch = {0: 40, 1: 30}
    sketch_size = len(sketch)
    element_count = sum(neighbor_sketch.values()) + 1
    decrement_count = 1
    privatization_offset = (decrement_count
                            - math.floor(element_count / (sketch_size + 1)))
    def input_generator2(sketch):
        _element_count = (element_count - 1 if sketch == neighbor_sketch
                          else element_count)
        _decrement_count = (decrement_count - 1 if sketch == neighbor_sketch
                          else decrement_count)
        return (sketch, sketch_size, epsilon, universe_size, _element_count,
                _decrement_count)
    title = "Privacy of pure privatization lowered counters"
    plot_privatization_distribution(
        title, repetitions, pmg.purely_privatize_misra_gries, sketch,
        neighbor_sketch, epsilon, delta, input_generator2,
        privatization_offset)
    plt.savefig("privacy_purely_privatize_lowered.png")

    title = "Privacy of pure privatization lowered counters by keys"
    plot_privatization_distribution(
        title, repetitions,
        lambda *args: list(pmg.purely_privatize_misra_gries(*args)), sketch,
        neighbor_sketch, epsilon, delta, input_generator2,
        privatization_offset)
    plt.savefig("privacy_purely_privatize_lowered_keys.png")


def test_privacy_privatize_merged():
    # repetitions = 10000
    repetitions = 1000
    merged = {0: 60, 1: 30, 2: 40}
    neighbor_merged = {0: 59, 1: 29, 2: 39}
    merged_size = len(merged)
    epsilon = 1
    delta = 1 / (sum(merged.values()) - 1 + 9) / 10

    def input_generator(merged):
        return merged, merged_size, epsilon, delta
    title = "Privacy of approximate privatization of merged"
    plot_privatization_distribution(
        title, repetitions, pmg.privatize_merged, merged, neighbor_merged,
        epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_merged.png")

    title = ("Privacy of approximate privatization of merged with loose "
             "threshold")
    with unittest.mock.patch("pmg.find_threshold",
                             new=pmg_alternatives.find_threshold_original):
        plot_privatization_distribution(
            title, repetitions, pmg.privatize_merged, merged, neighbor_merged,
            epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_merged_original_threshold.png")


def test_privacy_purely_privatize_merged():
    # repetitions = 20000
    repetitions = 2000
    merged = {0: 60, 1: 30, 2: 40}
    neighbor_merged = {0: 59, 1: 29, 2: 39}
    merged_size = len(merged)
    epsilon = 2
    delta = 0
    universe_size = 13

    def input_generator(merged):
        return merged, merged_size, epsilon, universe_size
    title = "Privacy of pure privatization of merged"
    plot_privatization_distribution(
        title, repetitions, pmg.purely_privatize_merged, merged,
        neighbor_merged, epsilon, delta, input_generator)
    plt.savefig("privacy_purely_privatize_merged.png")

    title = "Privacy of pure privatization of merged by keys"
    plot_privatization_distribution(
        title, repetitions,
        lambda *args: list(pmg.purely_privatize_merged(*args)), merged,
        neighbor_merged, epsilon, delta, input_generator)
    plt.savefig("privacy_purely_privatize_merged_keys.png")


def test_privacy_privatize_user_level():
    # repetitions = 10000
    repetitions = 1000
    sketch = {0: 60, 1: 30}
    neighbor_sketch = {0: 58, 1: 29}
    epsilon = 1
    delta = 1 / (sum(sketch.values()) - 1 + 9) / 10
    user_element_count = sum(sketch.values()) - sum(neighbor_sketch.values())

    def input_generator(sketch):
        return sketch, epsilon, delta, user_element_count
    title = "Privacy of user-level approximate privatization"
    plot_privatization_distribution(
        title, repetitions, pmg.privatize_user_level, sketch, neighbor_sketch,
        epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_user_level.png")

    title = "Privacy of user-level approximate privatization by keys"
    plot_privatization_distribution(
        title, repetitions,
        lambda *args: list(pmg.privatize_user_level(*args)), sketch,
        neighbor_sketch, epsilon, delta, input_generator)
    plt.savefig("privacy_privatize_user_level_keys.png")


def test_privacy_purely_privatize_user_level():
    # repetitions = 20000
    repetitions = 2000
    sketch = {0: 60, 1: 30}
    neighbor_sketch = {0: 58, 1: 29}
    sketch_size = len(sketch)
    epsilon = 2
    delta = 0
    universe_size = 12
    element_count = sum(sketch.values())
    decrement_count = 0
    user_element_count = sum(sketch.values()) - sum(neighbor_sketch.values())
    privatization_offset = (decrement_count
                            - math.floor(element_count / (sketch_size + 1)))

    def input_generator(sketch):
        _element_count = (element_count - user_element_count
                          if sketch == neighbor_sketch else element_count)
        return (sketch, sketch_size, epsilon, universe_size, _element_count,
                decrement_count, user_element_count)
    title = "Privacy of user-level pure privatization"
    plot_privatization_distribution(
        title, repetitions, pmg.purely_privatize_user_level, sketch,
        neighbor_sketch, epsilon, delta, input_generator, privatization_offset)
    plt.savefig("privacy_purely_privatize_user_level.png")

    title = "Privacy of user-level pure privatization by keys"
    plot_privatization_distribution(
        title, repetitions,
        lambda *args: list(pmg.purely_privatize_user_level(*args)), sketch,
        neighbor_sketch, epsilon, delta, input_generator, privatization_offset)
    plt.savefig("privacy_purely_privatize_user_level_keys.png")


def test_accuracy_privatize():
    repetitions = 10
    epsilon = 1
    delta = 1e-6
    # sketch_sizes = [10 * 2 ** i for i in range(14)]
    sketch_sizes = [10 * 2 ** i for i in range(8)]

    def input_generator(sketch_size):
        return {i: i % 50 for i in range(sketch_size)}, epsilon, delta
    def inaccuracy_generator(sketch_size, beta):
        max_inaccuracy = math.log((sketch_size + 1) * math.exp(epsilon)
                                  / ((math.exp(epsilon) + 1) * beta)) / epsilon
        min_inaccuracy = (- max_inaccuracy - 1
                          - 2 * math.log(6 * math.exp(epsilon)
                                         / ((math.exp(epsilon) + 1) * delta))
                          / epsilon)
        return min_inaccuracy, max_inaccuracy
    plt.clf()
    title = "Accuracy of approximate privatization"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Total inaccuracy")
    with unittest.mock.patch("pmg.find_threshold",
                             new=pmg_alternatives.find_threshold_original):
        plot_accuracy(title, "Loose threshold", repetitions,
                      pmg.privatize_misra_gries, sketch_sizes, input_generator,
                      inaccuracy_generator)
    plot_accuracy(title, "Numerical threshold (final)", repetitions,
                  pmg.privatize_misra_gries, sketch_sizes, input_generator,
                  inaccuracy_generator)
    plt.legend(loc="upper right")
    plt.savefig("accuracy_privatize.png")


def test_accuracy_purely_privatize():
    repetitions = 10
    epsilon = 1
    # sketch_sizes = [10 * 2 ** i for i in range(14)]
    sketch_sizes = [10 * 2 ** i for i in range(8)]
    universe_size = sketch_sizes[-1] * 10
    decrement_count = 0

    def input_generator(sketch_size):
        return ({i: i for i in range(sketch_size)}, sketch_size, epsilon,
                universe_size, sketch_size, decrement_count)
    plt.clf()
    title = "Accuracy of pure privatization"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Total inaccuracy")
    plot_accuracy(title, "", repetitions, pmg.purely_privatize_misra_gries,
                  sketch_sizes, input_generator)
    plt.savefig("accuracy_purely_privatize.png")

    sketch = {i: i for i in range(100)}
    sketch_size = len(sketch)
    universe_sizes = [sketch_size * 4 * 2 ** i for i in range(14)]
    # universe_sizes = [sketch_size * 4 * 2 ** i for i in range(8)]
    def input_generator2(universe_size):
        return (sketch, sketch_size, epsilon, universe_size, sketch_size,
                decrement_count)
    plt.clf()
    title = "Accuracy of pure privatization (universe size)"
    plt.title(title)
    plt.xlabel("Universe size")
    plt.xscale("log")
    plt.ylabel("Total inaccuracy")
    plt.yscale("log")
    plot_accuracy(title, "", repetitions, pmg.purely_privatize_misra_gries,
                  universe_sizes, input_generator2, sci_y=False)
    plt.savefig("accuracy_purely_privatize_universe.png")


def test_accuracy_privatize_merged():
    repetitions = 10
    epsilon = 1
    delta = 1e-6
    # sketch_sizes = [10 * 2 ** i for i in range(8)]
    sketch_sizes = [10 * 2 ** i for i in range(6)]

    def input_generator(sketch_size):
        return ({i: i * 100 for i in range(sketch_size)}, sketch_size, epsilon,
                delta)
    plt.clf()
    title = "Accuracy of approximate privatization of merged"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Total inaccuracy")
    with unittest.mock.patch("pmg.find_threshold",
                             new=pmg_alternatives.find_threshold_original):
        plot_accuracy(title, "Loose threshold", repetitions,
                      pmg.privatize_merged, sketch_sizes, input_generator)
    plot_accuracy(title, "Numerical threshold (final)", repetitions,
                  pmg.privatize_merged, sketch_sizes, input_generator)
    plt.legend(loc="upper right")
    plt.savefig("accuracy_privatize_merged.png")


def test_accuracy_purely_privatize_merged():
    repetitions = 10
    epsilon = 1
    # sketch_sizes = [10 * 2 ** i for i in range(12)]
    sketch_sizes = [10 * 2 ** i for i in range(8)]
    universe_size = sketch_sizes[-1] * 10

    def input_generator(sketch_size):
        return ({i: i for i in range(sketch_size)}, sketch_size, epsilon,
                universe_size)
    plt.clf()
    title = "Accuracy of pure privatization of merged"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Total inaccuracy")
    plot_accuracy(title, "", repetitions, pmg.purely_privatize_merged,
                  sketch_sizes, input_generator)
    plt.savefig("accuracy_purely_privatize_merged.png")

    sketch = {i: i for i in range(100)}
    sketch_size = len(sketch)
    universe_sizes = [sketch_size * 4 * 2 ** i for i in range(14)]
    # universe_sizes = [sketch_size * 4 * 2 ** i for i in range(8)]
    def input_generator2(universe_size):
        return (sketch, sketch_size, epsilon, universe_size)
    plt.clf()
    title = "Accuracy of pure privatization of merged (universe size)"
    plt.title(title)
    plt.xlabel("Universe size")
    plt.xscale("log")
    plt.ylabel("Total inaccuracy")
    plt.yscale("log")
    plot_accuracy(title, "", repetitions, pmg.purely_privatize_merged,
                  universe_sizes, input_generator2, sci_y=False)
    plt.savefig("accuracy_purely_privatize_merged_universe.png")


def test_accuracy_privatize_user_level():
    repetitions = 10
    epsilon = 1
    delta = 1e-6
    # sketch_sizes = [10 * 2 ** i for i in range(14)]
    sketch_sizes = [10 * 2 ** i for i in range(8)]
    user_element_count = 5

    def input_generator(sketch_size):
        return ({i: i for i in range(sketch_size)}, epsilon, delta,
                user_element_count)
    plt.clf()
    title = "Accuracy of user-level approximate privatization"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Total inaccuracy")
    plot_accuracy(title, "", repetitions, pmg.privatize_user_level,
                  sketch_sizes, input_generator)
    plt.savefig("accuracy_privatize_user_level.png")


def test_accuracy_purely_privatize_user_level():
    repetitions = 10
    epsilon = 1
    # sketch_sizes = [10 * 2 ** i for i in range(14)]
    sketch_sizes = [10 * 2 ** i for i in range(8)]
    universe_size = sketch_sizes[-1] * 10
    decrement_count = 0
    user_element_count = 5

    def input_generator(sketch_size):
        return ({i: i for i in range(sketch_size)}, sketch_size, epsilon,
                universe_size, sum(range(sketch_size)), decrement_count,
                user_element_count)

    plt.clf()
    title = "Accuracy of user-level pure privatization"
    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Total inaccuracy")
    plot_accuracy(title, "", repetitions, pmg.purely_privatize_user_level,
                  sketch_sizes, input_generator)
    plt.savefig("accuracy_purely_privatize_user_level.png")

    sketch = {i: i for i in range(100)}
    sketch_size = len(sketch)
    element_count = sum(range(sketch_size))
    universe_sizes = [sketch_size * 4 * 2 ** i for i in range(14)]
    # universe_sizes = [sketch_size * 4 * 2 ** i for i in range(8)]
    def input_generator2(universe_size):
        return (sketch, sketch_size, epsilon, universe_size, element_count,
                decrement_count, user_element_count)
    plt.clf()
    title = "Accuracy of user-level pure privatization (universe size)"
    plt.title(title)
    plt.xlabel("Universe size")
    plt.xscale("log")
    plt.ylabel("Total inaccuracy")
    plt.yscale("log")
    plot_accuracy(title, "", repetitions, pmg.purely_privatize_user_level,
                  universe_sizes, input_generator2, sci_y=False)
    plt.savefig("accuracy_purely_privatize_user_level_universe.png")


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


def test_accuracy():
    """Run the accuracy tests."""
    test_accuracy_privatize()
    test_accuracy_purely_privatize()
    test_accuracy_privatize_merged()
    test_accuracy_purely_privatize_merged()
    test_accuracy_privatize_user_level()
    test_accuracy_purely_privatize_user_level()


def main():
    """Run the evaluation of the private Misra-Gries sketching program."""
    print("RUNNING UNIT TESTS...")
    test()
    print("-" * 70)
    print("RUNNING BENCHMARKS...")
    benchmark()
    print("-" * 70)
    print("RUNNING STOCHASTIC PRIVACY TESTS...")
    test_privacy()
    print("-" * 70)
    print("RUNNING ACCURACY TESTS...")
    test_accuracy()


if __name__ == "__main__":
    main()
