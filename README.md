# Automated-Outlier-Detection-and-Estimation-of-Missing-Data

This repository contains the software for [Automated Outlier Detection and Estimation of Missing Data] which can be used for data imputation while minimizing the impact of outliers.
This software is associated with the paper 'Automated Outlier Detection and Estimation of Missing Data' by Jinwook Rhyu et al.

The software is performed in Python where `main_demonstration` and `main_validation` are the main functions for [Section 3. Demonstration] and [Section 4. Validation], respectively. The user may edit the parameters based on their dataset until line "### Step A-0: Preprocessing before Step A (Only use variables_mask and observations_mask)".

![alt text](https://github.com/JinwookRhyu/Automated-Outlier-Detection-and-Estimation-of-Missing-Data/blob/main/Process_diagram.png?raw=true)

# `Codes` folder
The major files under `Codes` folder are:
1. `Addmissingness`: Add missing patterns to the full dataset. Please refer to [Severson, K. A., Molaro, M. C., & Braatz, R. D. (2017). Principal component analysis of process datasets with missing values. Processes, 5(3), 38.] for more information.
2. `Algorithms`: Stores 9 imputation algorithms (MI, Alternating, SVDImpute, PCADA, PPCA, PPCA-M, BPCA, SVT, and ALM) described in [Section 2.3. Imputation algorithms for missing values (Step B)].
3. `Determine_A`: Determines the number of principal components based on cross-validation and calculates statistical metrics (e.g. T^2 and Q contributions, thresholds for each contribution, etc.).
4. `Fill_missing`: Iterates (a) data imputation and (b) determination of principal components until the number of principal components converges.
5. `Plot_dataset`: Generates plots where blue circles indicate normal data, cyan triangles indicate temporarily imputed missing values, red stars indicate detected outliers, green triangles indicate estimated missing values, and olive stars indicated replaced outliers.
6. `Preprocessing`: Preprocessing by (A0) use only the masked variables and observations, and (A1) temporarily impute missing values using either mean imputation, interpolation, or last observed values.
7. `main_demonstration`: The main code used in the [Section 3. Demonstration].
8. `main_validation`: The main code used in the [Section 4. Validation].

# `Codes_MATLAB` folder
The MATLAB version of this software, which is around 5-10 times faster than Python version, is located in `Codes_MATLAB` folder.

Reference for AddMissingness software: Severson, K.A., Molaro, M.C., and Braatz, R.D. Methods for applying principal component analysis to process datasets wiht missing values. Processes 2017, 5(3), 38. [http://web.mit.edu/braatzgroup/links.html]

Reference for BPCA algorithm: Oba, S., Sato, M., Takemasa, I., Monden, M., Matsubara, K., and Ishii, S. A Bayesian Missing value estimation method, Bioinformatics 19, pp.2088-2096 (2003). [http://ishiilab.jp/member/oba/tools/BPCAFill.html]

# `Dataset` folder
The `Dataset` folder contains the following two datasets:
1. `mAb_dataset_demonstration.xlsx`: The original dataset used in the the [Section 3. Demonstration].
2. `mAb_dataset_validation.xlsx`: The preprocessed dataset used in the the [Section 4. Validation].

Please contact Richard Braatz at braatz@mit.edu for any inquiry. 


# Acknowledgement
This study was supported by the U.S. Food and Drug Administration, Contract No. 75F40121C00090. Any opinions,
findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily
reflect the views of the financial sponsor. MIT thanks Sartorius Stedim Cellca GMBH for the generous support of the adalibumab-producing CHO cell line.
