# rail_nncflag

Implementation of the neural-net-based photo-z quality flag code originally written by Adam Broussard and updated by Irene Moskowitz, adapted for use with RAIL.  This code implements only the basic version, which takes in a set of photometric bands, a point estimate redshift, and true redshifts for a training file, and uses as features the reference band magnitude and the galaxy colors to train a set of neural nets.  These models are applied to the same features in a test set, and the average of the quality scores is returned as a single number ranging from 0(likely bad) to 1(likely good).

The original code read in "zconf", a measure of how peaked the PDF distribution is, but did not use it.  I added an option `include_odds` that calculates ODDS, the integral of the amount of posterior probability in an interval +/- 0.06*(1+zmode) around the PDF mode.  ODDS measures the "peakiness" of the distribution, narrow unimodal PDFs will have a high ODDS value, while multimodal and broad unimodal PDFs will have lower ODDS values.  Setting `include_odds` to `True` will include ODDS as an additional parameter in the neural net training, and may improve reliability.

Currently, I have the code set up using the inform/estimate Estimator base classes.  This might change in future versions.

