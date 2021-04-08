# cross-entropy-for-combinatorics
Code accompanying the manuscript "Constructions in combinatorics via neural networks" by A Z Wagner

The cem_binary_conj21.py file contains the solution to Conjecture 2.1 in the paper. It does not use numba for speed which makes the code rather slow, but it is enough to find the counterexample given in the paper.

The cem_binary_conj23_with_numba.py file contains the solution to Conjecture 2.3. It demonstrates the use of numba to speed up the calculation of the reward. Both of these work on tensorflow version 1.14.0, python version 3.6.3, keras version 2.3.1.

The dqn/ folder contains an unsuccessful attempt at a solution of Conjecture 2.1 using Dueling Deep Q Networks. The code for the main algorithm was mostly taken from here: https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code . It might eventually find the a counterexample, but it performed significantly slower on my computer than the simple cross-entropy method solution. Install the environment first by going into the main folder and writing pip install -e gym-conj21, and make sure you have created models, plots, and best_species folders before running it.
