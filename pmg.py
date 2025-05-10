#!/usr/bin/env python3
"""Differentially private Misra-Gries in practice.

This module allows creating and merging differentially private Misra-Gries
sketches. It can be run from the command-line with the relevant arguments that
are shown in a usage message when running without arguments, where the output
sketches are then printed and (for the non-private sketches) optionally saved
to a file. The functions can also be called from other modules, where the
functions allow creating the non-private sketches and merged sketches, and
these can be either approximately or purely privatized.
"""


from collections.abc import Callable, Iterable, Sequence
import json
import math
import random
import sys

from numpy.random import binomial


RANDOM = random.SystemRandom()


def misra_gries(stream: Iterable[int],
                sketch_size: int,
                ) -> tuple[dict[int, int], int, int]:
    """Calculate the Misra-Gries sketch of the given stream.

    The sketch will be of up to the given size, where any keys with counters of
    zero are kept. Only integer elements >= 0 are considered valid keys.

    Args:
        stream: An iterable of integer elements to be counted for the sketch.
        sketch_size: The maximum size of the sketch.

    Returns:
        A tuple with the Misra-Gries sketch, along with the total count of
        (valid) elements considered and count of decrements performed during
        the calculation.
    """
    # Initialize the sketch with invalid keys with zero counters, allowing the
    # algorithm to also insert keys in the same way in the beginning.
    sketch = {key: 0 for key in range(-sketch_size, 0)}
    zero_group = list(range(-sketch_size, 0))
    zero_pointer = 0
    element_count = 0
    decrement_count = 0

    def decrement_all() -> None:
        """Decrement all counters while updating the zero group."""
        nonlocal zero_group, zero_pointer, decrement_count
        decrement_count += 1
        zero_group = []
        for key in sketch:
            sketch[key] -= 1
            if sketch[key] == 0:
                zero_group.append(key)

        # Sort the zero counters so they can be replaced in key order.
        zero_group.sort()
        zero_pointer = 0

    def insert_key(key: int) -> None:
        """Insert the given key at the smallest key with a zero count."""
        nonlocal zero_pointer
        while True:
            removed_key = zero_group[zero_pointer]
            zero_pointer += 1
            if sketch[removed_key] == 0:
                break
            if zero_pointer == len(zero_group):
                # We have no more zero counters, so we must decrement instead.
                decrement_all()
                return
        del sketch[removed_key]
        sketch[key] = 1

    # Perform the Misra-Gries algorithm for counting all the stream elements.
    for element in stream:
        if element < 0:
            continue
        element_count += 1
        if element in sketch:
            sketch[element] += 1
        elif zero_pointer == len(zero_group):
            decrement_all()
        else:
            insert_key(element)

    # Only include valid keys, while sorting to make the output more readable.
    final_sketch = {}
    for key in sorted(sketch):
        if key >= 0:
            final_sketch[key] = sketch[key]

    return final_sketch, element_count, decrement_count


def privatize_misra_gries(sketch: dict[int, int],
                          epsilon: float,
                          delta: float,
                          sensitivity: int = 1,
                          threshold: float = -1,
                          add_global_noise: bool = True,
                          ) -> dict[int, int]:
    """Approximately privatize the given Misra-Gries sketch.

    By default, a given sketch as outputted by the misra_gries() function is
    privatized according to its global sensitivity of 1 by exploiting its
    structure through adding global two-sided geometric noise as well as
    individual noise to each counter.

    Args:
        sketch: The Misra-Gries sketch to privatize.
        epsilon: The epsilon parameter for the approximate privacy.
        delta: The delta parameter for the approximate privacy.
        sensitivity: The global L1 sensitivity of the given sketch.
        threshold: The threshold for counters to be included in the private
            sketch. If -1, the default threshold is calculated based on delta.
        add_global_noise: Whether or not to add global noise as well.

    Returns:
        The approximately privatized Misra-Gries sketch.
    """
    if threshold == -1:
        threshold = find_threshold(epsilon, delta, 1)
    two_sided_geometric = create_two_sided_geometric(epsilon, sensitivity)
    global_noise = two_sided_geometric() if add_global_noise else 0

    # Include all counters that are above the threshold after adding noise,
    # while sorting the counters to ensure privacy.
    private_sketch = {}
    for key in sorted(sketch):
        counter = sketch[key] + global_noise + two_sided_geometric()
        if counter >= threshold:
            private_sketch[key] = counter

    return private_sketch


def purely_privatize_misra_gries(sketch: dict[int, int],
                                 sketch_size: int,
                                 epsilon: float,
                                 universe_size: int,
                                 element_count: int,
                                 decrement_count: int,
                                 sensitivity: int = 2,
                                 offset_counters: bool = True,
                                 ) -> dict[int, int]:
    """Purely privatize the given Misra-Gries sketch.

    The given sketch should consist of integer keys >= 0 and < universe_size.
    By default, a sketch as outputted by the misra_gries() function is
    privatized by first post-processing it through offsetting all counters,
    ensuring the global L1 sensitivity is only 2.

    Args:
        sketch: The Misra-Gries sketch to privatize.
        sketch_size: The maximum size that was desired for the given sketch.
        epsilon: The epsilon parameter for the pure privacy.
        universe_size: The size of the element universe.
        element_count: The count of (valid) elements that were seen in the
            input stream. If offset_counters is False, this value is unused.
        decrement_count: The count of decrements performed when calculating the
            given sketch. If offset_counters is False, this value is unused.
        sensitivity: The global L1 sensitivity of the given sketch.
        offset_counters: Whether or not to first perform the post-processing.

    Returns:
        The purely privatized Misra-Gries sketch.
    """
    offset = (decrement_count - math.floor(element_count / (sketch_size + 1))
              if offset_counters else 0)
    threshold = math.ceil(
        math.log((1 + math.exp(-epsilon / sensitivity)) * sketch_size
                 / universe_size)
        / math.log(math.exp(-epsilon / sensitivity)))
    geometric = create_geometric(epsilon, sensitivity)
    two_sided_geometric = create_two_sided_geometric(epsilon, sensitivity)

    # Include all noisy counters above the calculated necessary threshold.
    noisy_sketch = {}
    for key in sketch:
        counter = sketch[key] + offset + two_sided_geometric()
        if counter >= threshold:
            noisy_sketch[key] = counter

    # Upgrade a random amount of zeros, selected with rejection sampling, to be
    # included with noise above the threshold.
    upgrade_count = binomial(universe_size, sketch_size / universe_size)
    while upgrade_count > 0:
        key = RANDOM.randrange(0, universe_size)
        if key not in noisy_sketch:
            noisy_sketch[key] = threshold + geometric()
            upgrade_count -= 1

    # Select the top sketch_size noisy counters, and sort them for readability.
    top_k = sorted(noisy_sketch.items(),
                   key=lambda item: item[1])[-sketch_size:]
    private_sketch = dict(sorted(top_k))

    return private_sketch


def merge(sketches: Sequence[dict[int, int]],
          sketch_size: int,
          ) -> dict[int, int]:
    """Calculate the merged Misra-Gries sketch from the given sketches.

    The merged sketch will at most be of the given size, but the given sketches
    can be of any size.

    Args:
        sketch: A sequence of Misra-Gries sketches to merge.
        sketch_size: The maximum size of the merged sketch.

    Returns:
        The merged Misra-Gries sketch."
    """
    merged = sketches[0]
    for sketch in sketches[1:]:
        summed_sketch = merged
        for key in sketch:
            if key in summed_sketch:
                summed_sketch[key] += sketch[key]
            else:
                summed_sketch[key] = sketch[key]

        if len(summed_sketch) > sketch_size:
            offset = sorted(summed_sketch.items(),
                            key=lambda item: item[1])[-(sketch_size + 1)][1]
            for key in summed_sketch:
                if summed_sketch[key] > offset:
                    merged[key] = summed_sketch[key] - offset
        else:
            merged = dict(summed_sketch)

    return merged


def privatize_merged(merged: dict[int, int],
                     sketch_size: int,
                     epsilon: float,
                     delta: float,
                     ) -> dict[int, int]:
    """Approximately privatize the given merged Misra-Gries sketch.

    This simply uses the privatize_misra_gries() function with a higher global
    sensitivity and threshold.

    Args:
        merged: The merged Misra-Gries sketch to privatize.
        sketch_size: The maximum size that was desired for the given merged
            sketch, which is used as the global sensitivity.
        epsilon: The epsilon parameter for the approximate privacy.
        delta: The delta parameter for the approximate privacy, which is used
            for selecting the threshold.

    Returns:
        The approximately privatized merged Misra-Gries sketch.
    """
    threshold = find_threshold(epsilon, delta, sketch_size, sketch_size)
    return privatize_misra_gries(merged, epsilon, delta, sketch_size,
                                 threshold, False)


def purely_privatize_merged(merged: dict[int, int],
                            sketch_size: int,
                            epsilon: float,
                            universe_size: int,
                            ) -> dict[int, int]:
    """Purely privatize the given merged Misra-Gries sketch.

    The merged sketch should consist of integer keys >= 0 and < universe_size.
    This simply uses the purely_privatize_misra_gries() function with a higher
    global sensitivity and without performing the post-processing.

    Args:
        merged: The merged Misra-Gries sketch to privatize.
        sketch_size: The maximum size that was desired for the given merged
            sketch, which is used as the global sensitivity.
        epsilon: The epsilon parameter for the pure privacy.
        universe_size: The size of the element universe.

    Returns:
        The purely privatized merged Misra-Gries sketch.
    """
    return purely_privatize_misra_gries(merged, sketch_size, epsilon,
                                        universe_size, None, None, sketch_size,
                                        False)


def privatize_user_level(sketch: dict[int, int],
                         epsilon: float,
                         delta: float,
                         user_element_count: int,
                         ) -> dict[int, int]:
    """User-level approximately privatize the given Misra-Gries sketch.

    Each user can contribute up to user_element_count elements to the sketch.
    This simply uses the privatize_misra_gries() function with scaled values of
    epsilon and delta.

    Args:
        sketch: The Misra-Gries sketch to privatize.
        epsilon: The epsilon parameter for the approximate privacy.
        delta: The delta parameter for the approximate privacy, which is used
            for selecting the threshold.
        user_element_count: The maximum amount of elements (not necessarily
            distinct) that each user can have contributed to the input stream.

    Returns:
        The user-level approximately privatized Misra-Gries sketch.
    """
    scaled_epsilon = epsilon / user_element_count
    scaled_delta = delta / (user_element_count * math.exp(epsilon))
    return privatize_misra_gries(sketch, scaled_epsilon, scaled_delta)


def purely_privatize_user_level(sketch: dict[int, int],
                                sketch_size: int,
                                epsilon: float,
                                universe_size: int,
                                element_count: int,
                                decrement_count: int,
                                user_element_count: int,
                                ) -> dict[int, int]:
    """User-level purely privatize the given Misra-Gries sketch.

    The sketch should consist of integer keys >= 0 and < universe_size. This
    simply uses the purely_privatize_misra_gries() function with a higher
    global sensitivity.

    Args:
        sketch: The Misra-Gries sketch to privatize.
        sketch_size: The maximum size that was desired for the given sketch.
        epsilon: The epsilon parameter for the pure privacy.
        universe_size: The size of the element universe.
        element_count: The count of (valid) elements that were seen in the
            input stream.
        decrement_count: The count of decrements performed when calculating the
            given sketch.
        user_element_count: The maximum amount of elements (not necessarily
            distinct) that each user can have contributed to the input stream.

    Returns:
        The user-level purely privatized Misra-Gries sketch.
    """
    return purely_privatize_misra_gries(sketch, sketch_size, epsilon,
                                        universe_size, element_count,
                                        decrement_count, user_element_count)


def create_geometric(epsilon: float,
                     sensitivity: float,
                     ) -> Callable[[], int]:
    """Create a sampling function for the geometric distribution.

    Args:
        epsilon: The privacy epsilon parameter for the desired noise.
        sensitivity: The global L1 sensitivity for the desired noise.

    Returns:
        A function for sampling from the geometric distribution that can be
        used for a privacy mechanism with the given parameters.
    """
    log_alpha = math.log(math.exp(-epsilon / sensitivity))
    return lambda: math.floor(math.log(1 - RANDOM.random()) / log_alpha)


def create_two_sided_geometric(epsilon: float,
                               sensitivity: float,
                               ) -> Callable[[], int]:
    """Create a sampling function for the two-sided geometric distribution.

    Args:
        epsilon: The privacy epsilon parameter for the desired noise.
        sensitivity: The global L1 sensitivity for the desired noise.

    Returns:
        A function for sampling from the two-sided geometric distribution that
        can be used for a privacy mechanism with the given parameters.
    """
    geometric = create_geometric(epsilon, sensitivity)
    return lambda: geometric() - geometric()


def find_threshold(epsilon: float,
                   delta: float,
                   sensitivity: float = 1,
                   max_unique_keys = 2,
                   ) -> int:
    """TODO:"""
    def pmf(j: int):
        # Return the probability that a two-sided geometric variable is equal
        # to any integer j.
        return ((1 - math.exp(-epsilon / sensitivity))
                / (1 + math.exp(-epsilon / sensitivity))
                * math.exp(-epsilon * abs(j) / sensitivity))

    def cdf(j: int):
        # Return the probability that a two-sided geometric variable is greater
        # than or equal to any integer j.
        j2 = -j + 1 if j < 0 else j
        probability = (math.exp((-j2 + 1) * (epsilon / sensitivity))
                       / (math.exp(epsilon / sensitivity) + 1))
        return 1 - probability if j < 0 else probability

    probability_goal = delta / 2
    probability_tolerance = delta / 1e9

    threshold = 1
    min_threshold = 1
    max_threshold = -1
    while min_threshold != max_threshold:
        total_probability = 0
        for global_noise_change in [1, -1]:
            global_noise = global_noise_change
            probability = 1
            while (total_probability < probability_goal
                   and probability > probability_tolerance):
                global_probability = pmf(global_noise)
                local_probability = cdf(threshold - global_noise)
                probability = (global_probability
                               * (1 - (1 - local_probability)
                                      ** max_unique_keys))
                total_probability += probability
                global_noise += global_noise_change

        if max_threshold != -1:
            if total_probability > probability_goal:
                min_threshold = threshold + 1
            else:
                max_threshold = threshold
            threshold = (min_threshold + max_threshold) // 2
        elif total_probability <= probability_goal:
            max_threshold = threshold
            threshold = (min_threshold + max_threshold) // 2
        else:
            min_threshold = threshold + 1
            threshold *= 2

    return threshold


def create_sketch() -> None:
    """Create a Misra-Gries sketch according to the command-line arguments.

    The sketch is created by streaming from a file given as a command-line
    argument. Both the non-private and private sketches are printed as output,
    with the non-private sketch optionally being saved as a JSON file. If the
    delta argument is 0, the functions for purely privatizing the sketch are
    used, and otherwise it is approximately privatized.
    """
    sketch_size = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    delta = float(sys.argv[3])
    if delta > 0:
        file = sys.argv[4]
    else:
        universe_size = int(sys.argv[4])
        file = sys.argv[5]

    # Create the sketch from the file stream.
    with open(file, encoding="utf8") as stream:
        stream = map(int, stream)
        sketch, element_count, decrement_count = misra_gries(stream,
                                                             sketch_size)

    # Privatize the sketch.
    if delta > 0:
        private_sketch = privatize_misra_gries(sketch, epsilon, delta)
    else:
        private_sketch = purely_privatize_misra_gries(
            sketch, sketch_size, epsilon, element_count, decrement_count,
            universe_size)

    # Output the sketch.
    print("Sketch        :", sketch)
    print("Private sketch:", private_sketch)
    if ((delta > 0 and len(sys.argv) >= 6) or len(sys.argv) >= 7):
        output_file = sys.argv[5] if delta > 0 else sys.argv[6]
        with open(output_file, "w", encoding="utf8") as output:
            json.dump(sketch, output)


def merge_sketches() -> None:
    """Merge Misra-Gries sketches according to the command-line arguments.

    The merged sketch is created by first reading into memory all the sketches
    from JSON files that were given as command-line arguments. The merged
    sketch will at most be of the given size, but the given sketches can be of
    any size. If the delta argument is 0, the functions for purely privatizing
    the merged sketch are used, and otherwise it is approximately privatized.
    """
    sketch_size = int(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    if delta > 0:
        sketch_files = sys.argv[5:]
    else:
        universe_size = int(sys.argv[5])
        sketch_files = sys.argv[6:]

    # Merge the sketches from the files.
    sketches = []
    for file in sketch_files:
        with open(file, encoding="utf8") as input_:
            sketches.append({int(key): counter for key, counter
                             in json.load(input_).items()})
    merged = merge(sketches, sketch_size)

    # Privatize the merged sketch.
    if delta > 0:
        private_merged = privatize_merged(merged, sketch_size, epsilon, delta)
    else:
        private_merged = purely_privatize_merged(merged, sketch_size, epsilon,
                                                 universe_size)

    # Output the merged sketch.
    print("Merged        :", merged)
    print("Private merged:", private_merged)


def create_user_level_sketch() -> None:
    """Create a user-level private Misra-Gries sketches according to the
    command-line arguments.

    The sketch is created by streaming from a file containing a flattened
    version of the input stream given as a command-line argument. Both the
    non-private and user-level private sketches are printed as output. If the
    delta argument is 0, the functions for purely privatizing the sketch are
    used, and otherwise it is approximately privatized.
    """
    sketch_size = int(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    user_element_count = int(sys.argv[5])
    if delta > 0:
        file = sys.argv[6]
    else:
        universe_size = int(sys.argv[6])
        file = sys.argv[7]

    # Create the sketch from the file stream.
    with open(file, encoding="utf8") as stream:
        stream = map(int, stream)
        sketch, element_count, decrement_count = misra_gries(stream,
                                                             sketch_size)

    # User-level privatize the sketch.
    if delta > 0:
        private_sketch = privatize_user_level(sketch, epsilon, delta,
                                              user_element_count)
    else:
        private_sketch = purely_privatize_user_level(sketch, sketch_size,
            epsilon, universe_size, element_count, decrement_count,
            user_element_count)

    # Output the user-level private sketch.
    print("Sketch                   :", sketch)
    print("User-level private sketch:", private_sketch)


def main() -> None:
    """Run the program for creating and merging private Misra-Gries sketches.

    If too few command-line arguments are given, a usage message is printed.
    Otherwise, the functions for creating and merging differentially private
    Misra-Gries sketches are called.
    """
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
        print("  Create a user-level (epsilon, delta)-private sketch:")
        print(f"    {sys.argv[0]} userlevel <sketch size> <epsilon> <delta> "
              "<user element count> <stream file>")
        print("  Create a user-level (epsilon, 0)-private sketch:")
        print(f"    {sys.argv[0]} userlevel <sketch size> <epsilon> 0 "
              "<user element count> <universe size> <sketch file>")
        return

    match sys.argv[1]:
        case "merge":
            merge_sketches()
        case "userlevel":
            create_user_level_sketch()
        case _:
            create_sketch()


if __name__ == "__main__":
    main()
