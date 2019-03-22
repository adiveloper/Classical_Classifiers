# Classical_Classifiers
## LDA_DLDA_NMC.ipynb
    shows a basic classification of 10 points using widely used classifiers(ANOVA) 
    The were used before machine learning(neural network) was popular and more coommonly used in statistics and pattern recognition. 
    To understand the codes a basic understanding of the foolowing is reuired:

## 1) LDA: 
      (https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
## 2) DLDA:
     (similar to LDA co-variance matrix Diagonalized)
## 3) NMC: 
      Implementation of the nearest mean classifier modeled.
      Classes are modeled as gaussians with equal, spherical covariance matrices.
      The optimal covariance matrix and means for the classes are found using maximum likelihood, 
      which, in this case, has a closed form solution.
      To get true nearest mean classification, set prior as a matrix with equal probabilty for all classes, i.e. matrix(0.5,2).
      
      

## Error_convergence_LDA_DLDA_NMC.ipynb
      This is a proof that the classification error reduces with the increase in sample data 
