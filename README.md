# cross-entropy-for-combinatorics
Code accompanying the manuscript "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner

The cem_binary_conj21_without_numba.py file contains the solution to Conjecture 2.1 in the paper. It does not use numba for speed and clarity which makes the code rather slow, but it is enough to find the counterexample given in the paper.

The cem_binary_conj23_with_numba.py file contains the solution to Conjecture 2.3. It uses numba to speed up the calculation of the reward.

This code works on tensorflow version 1.14.0, python version 3.6.3, keras version 2.3.1.
