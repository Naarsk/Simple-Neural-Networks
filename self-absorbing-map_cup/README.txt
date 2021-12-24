DESCRIPTION
This code:
-generates random points (random.c) on the surface of a unit sphere, uniformly distributed along the azimuthal angle and normally distributed along the polar angle around theta=0, 
-prints the coordinates x,y,z of the points (in var.h and pion.dat), 
-creates and trains a 2-dimensional self absorbing map on the inputs (cup.c), 
-visualizes the results with Mathematica (cup.nb), including the creation of an animated sequence (gif) of the network over the inputs as the training progresses.

INSTRUCTION
To do the above:
1-Compile random.c using gcc 
2-Execute the compiled file. This generates the random points and prints them in 2 files: 
	- var.h 	: point coordinates to be used as input for c.
	- poin.dat	: point coordinates to be used as input for mathematica.
3-Compile and execute cup.c. This generates a file containing nodes coordinates. Be careful -- it could take time.
4-Open cup.nb with Wolfram Mathematica and execute the notebook to create the animation file cup.gif.