# cross-entropy-for-combinatorics
Code accompanying the arXiv version of the manuscript "Constructions in combinatorics via neural networks" by A Z Wagner
https://arxiv.org/abs/2104.14516


### Software requirements

- Tensorflow version 1.14.0
- Python version 3.6.3
- Keras version 2.3.1

### Demos

The cem_binary_conj21.py file contains the solution to Conjecture 2.1 in the paper. The code is very simple and rather slow, but it is enough to find the counterexample given in the paper within a day.

The cem_binary_conj23_with_numba.py file contains the solution to Conjecture 2.3. It uses the numba package to speed up the calculation of the reward. It will find a graph similar to Figure 5 in the arXiv paper, within a few days.

### Installation and usage

Install the versions of Tensorflow, Python, and Keras as above. Note that having higher versions of Tensorflow and Python might drastically reduce the performance of the code.

Download the code_template.py file. Change the variable `DECISIONS` to the number of binary decisions in your problem, and choose the hyperparameters. Next, fill in the reward function for your problem in the `calc_score` function. The input to this function is a 0-1 list of length `DECISIONS`, representing your graph or other object you have created. See demos for examples. Run the program simply with the `python code_template.py` command, no installation is necessary.

### Output

During runtime, the program will display the scores of the best constructions in the current iteration. All the information is also saved every 20 iterations into files, both in pickle and txt formats, in the same folder as the .py file.




