DESCRIPTION:
This code creates a 3-layered feed-forward neural network and trains it on a set of 100 examples 
(regarding the likeliness of a couple to divorce based on the answer given to some relationship questions -- more info at:
https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set),
and tests the generalization on 70 more examples. It produces a file plot.dat that contains three columns of data: (iteration number, error on the learnset, error on the testset) that can be used for result visualization.

INSTRUCTIONS:
-Compile and execute divorce.c using gcc,
-Plot the data in plot.dat, ie using GNUplot.
