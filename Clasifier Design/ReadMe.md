 In this part we will apply classiﬁer design, error estimation, and feature selection techniques to a gene expression data set from 
 the following cancer classiﬁcation study: 
 van de Vijver,M.J., He,Y.D., van’t Veer, L.J., et al. (2002), “A gene-expression signature as a predictor of survival in breast cancer.” New Eng. J. Med., 347, 1999-2009.
 This paper analyzes gene expression in breast tumor biopsies from 295 patients.
 The authors performed feature selection to obtain 70 genes; 
 hence, the full data matrix is 70×295. This is a retrospective study, meaning that the patients were tracked over the years and their 
 outcomes recorded. Using this clinical information, the authors labeled the patients into two classes: the “good’ prognosis” group were 
 disease-free for at least ﬁve years after ﬁrst treatment, whereas the “bad prognosis” group developed distant methatasis within the ﬁrst 
 ﬁve years. Of the 295 patients, 216 belong to the “good-prognosis” class, whereas the remaining 79 belong to the “poor-prognosis” class. 
 The gene expression data was randomly divided into a training data set (containing 80 points, with 40 points from each class) and testing 
 data set (containing the remaining 215 points). The latter will be used for test-set error estimation of the true classiﬁcation error. 
 The tab-delimited ﬁles are available on the TAMU Google Drive at http://bit.ly/2jaGCkg. The ﬁrst row contains the gene symbol names, whereas
 the ﬁrst column contains the patient ID. The last column contains the label (1 = good prognosis, 0 = poor prognosis). We are going to search
 for gene feature sets and design classiﬁers that best discriminate the two prognosis classes on the training data, and use the testing data
 to determine their accuracy. We will consider the following classiﬁcation rules:
 • Linear SVM, C = 1. 
 • Nonlinear SVM with RBF kernel, C = 10. 
 • Neural network with 2 hidden layers of 5 neurons each and logistic nonlinearities. 
