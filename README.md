# Automated-Outlier-Detection-and-Estimation-of-Missing-Data

This repository contains the software for [Automated Outlier Detection and Estimation of Missing Data] which can be used for data imputation while minimizing the impact of outliers.
This software is associated with the paper 'Automated Outlier Detection and Estimation of Missing Data' by Jinwook Rhyu et al.

The software is performed in Python where the 

https://github.com/JinwookRhyu/Automated-Outlier-Detection-and-Estimation-of-Missing-Data/blob/main/Process_diagram.jpg?raw=true

The major files under `Automated_Outlier_Detection_and_Estimation_of_Missing_Data` are:
1. `Addmissingness`: Add missing patterns to the full dataset. Please refer to [Severson, K. A., Molaro, M. C., & Braatz, R. D. (2017). Principal component analysis of process datasets with missing values. Processes, 5(3), 38.] for more information.
2. `Algorithms`: Stores 9 imputation algorithms (MI, Alternating, SVDImpute, PCADA, PPCA, PPCA-M, BPCA, SVT, and ALM) described in [Section 2.3. Imputation algorithms for missing values (Step B)].
3. `Determine_A`: Determines the number of principal components based on cross-validation and calculates statistical metrics (e.g. T^2 and Q contributions, thresholds for each contribution, etc.).
4. `Fill_missing`: Iterates (a) data imputation and (b) determination of principal components until the number of principal components converges.
5. `Plot_dataset`: Generates plots where blue circles indicate normal data, cyan triangles indicate temporarily imputed missing values, red stars indicate detected outliers, green triangles indicate estimated missing values, and olive stars indicated replaced outliers.
6. `Preprocessing`: Preprocessing by (A0) use only the masked variables and observations, and (A1) temporarily impute missing values using either mean imputation, interpolation, or last observed values.
7. `main_demonstration`: The main code used in the [Section 3. Demonstration]
8. `main_validation`: The main code used in the [Section 4. Validation]

Please contact Richard Braatz at braatz@mit.edu for any inquiry. 
