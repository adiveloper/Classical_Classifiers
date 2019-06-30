
We will use the Carnegie Mellon University Ultrahigh Carbon Steel (CMU-UHCS) dataset in
B. DeCost, T. Francis and E. Holm (2017), “Exploring the microstructure manifold: image texture representations applied to ultrahigh carbon steel microstructures.” arXiv:1702.01117v2.

The data set is available on the TAMU Google Drive at http://bit.ly/2jaGCkg. 
There are three ﬁles: a ZIP ﬁle containing the raw images and two excel ﬁles containing the labels and sample preparation information.
Please read DeCost’s paper to learn more about the data set. We will classify the micrographs according to primary microconstituent. 
There are a total of seven diﬀerent labels, corresponding to diﬀerent phases of steel resulting from diﬀerent thermal processing.
In this assignment, we will use only the spheroidite (374 micrographs), network (212 micrographs), and pearlite (124) micrographs.
The training data will be the ﬁrst 100 data points in the spherodite, network, and pearlite categories.
The remaining data points from these categories will compose the test sets (more below).
The classiﬁcation rule to be used is a Radial Basis Function (RBF) nonlinear SVM classiﬁcation rule.
We will use a one-vs-one approach to deal with the multiple labels, where each of the 3 classiﬁcation problems
for each pair of labels are carried out.
Given a new image, each of the 3 classiﬁers is applied and then a vote is taken to achieve a consensus for the most often predicted label.
If there is a 3-way tie, then no label is output. 
