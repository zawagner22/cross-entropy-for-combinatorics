# cross-entropy-for-combinatorics
Code accompanying the manuscript "Constructions in combinatorics via neural networks" by A Z Wagner

The cem_binary_conj21.py file contains the solution to Conjecture 2.1 in the paper. It does not use numba for speed which makes the code rather slow, but it is enough to find the counterexample given in the paper.

The cem_binary_conj23_with_numba.py file contains the solution to Conjecture 2.3. It demonstrates the use of numba to speed up the calculation of the reward. Both of these work on Tensorflow version 1.14.0, Python version 3.6.3, Keras version 2.3.1.
