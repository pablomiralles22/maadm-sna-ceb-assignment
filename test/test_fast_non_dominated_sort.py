import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from nsga2_utils import fast_non_dominated_sort

def test_fast_non_dominated_sort_empty_population():
    population = []
    assert fast_non_dominated_sort(population) == [[]]

def test_fast_non_dominated_sort_single_individual():
    population = [((1, 2), [0]),]
    assert fast_non_dominated_sort(population) == [[0]]

def test_fast_non_dominated_sort_two_dominated_individuals():
    population = [
        ((1, 2), [0]),
        ((3, 4), [1]),
    ]
    assert fast_non_dominated_sort(population) == [[1], [0]]

def test_fast_non_dominated_sort_two_dominated_individuals_reverse_order():
    population = [
        ((3, 4), [0]),
        ((1, 2), [1]),
    ]
    assert fast_non_dominated_sort(population) == [[0], [1]]

def test_fast_non_dominated_sort_two_non_dominated_individuals():
    population = [
        ((3, 2), [0]),
        ((1, 3), [1]),
    ]
    assert fast_non_dominated_sort(population) == [[0, 1]]

def test_fast_non_dominated_sort_one_dominates_other_two():
    population = [
        ((5, 6), [0]),
        ((3, 4), [1]),
        ((4, 3), [2]),
    ]
    assert fast_non_dominated_sort(population) == [[0], [1, 2]]

def test_fast_non_dominated_sort_two_non_dominated_and_one_dominated():
    population = [
        ((1, 2), [0]),
        ((3, 1), [1]),
        ((1, 0), [2]),
    ]
    assert fast_non_dominated_sort(population) == [[0, 1], [2]]

def test_fast_non_dominated_sort_all_non_dominated():
    population = [
        ((1, 4), [0]),
        ((2, 3), [1]),
        ((3, 2), [2]),
    ]
    assert fast_non_dominated_sort(population) == [[0, 1, 2]]

def test_fast_non_dominated_sort_two_non_dominated_and_two_dominated():
    population = [
        ((2, 3), [0]),
        ((3, 2), [1]),
        ((0, 1), [2]),
        ((1, 0), [3]),
    ]
    assert fast_non_dominated_sort(population) == [[0, 1], [2, 3]]

def test_fast_non_dominated_sort_three_non_dominated_and_one_dominated():
    population = [
        ((1, 4), [0]),
        ((2, 3), [1]),
        ((3, 2), [2]),
        ((0, 1), [3]),
    ]
    assert fast_non_dominated_sort(population) == [[0, 1, 2], [3]]

def test_fast_non_dominated_sort_all_non_dominated():
    population = [
        ((1, 5), [0]),
        ((2, 4), [1]),
        ((3, 3), [2]),
        ((4, 2), [3]),
    ]
    assert fast_non_dominated_sort(population) == [[0, 1, 2, 3]]

def test_fast_non_dominated_sort_non_strict_domination():
    population = [
        ((1, 5), [0]),
        ((1, 5), [1]),
    ]
    assert fast_non_dominated_sort(population) == [[0, 1]]

if __name__ == "__main__":
    pytest.main()