# cross-entropy-for-combinatorics
Code accompanying the manuscript "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner

Please keep in mind that I am far from being an expert in reinforcement learning. 
If you know what you are doing, you might be better off writing your own code.

This code works on tensorflow version 1.14.0 and python version 3.6.3
It mysteriously breaks on other versions of python.
For later versions of tensorflow there seems to be a massive overhead in the predict function for some reason, and/or it produces mysterious errors.
Debugging these was way above my skill level.
If the code doesn't work, make sure you are using these versions of tf and python.

I used keras version 2.3.1, not sure if this is important, but I recommend this just to be safe.
