import numpy as np
import copy

def addMissingness(X, miss_type, level, tol):
    # Function to add missingness at a specfied level to the input dataset
    ###########################################################################
    #                                 Inputs                                  #
    ###########################################################################
    # x - data set organized such that the rows are samples (or timepoints) and
    # the columns are measurements x may already contain missing elements,
    # which should be represented by NaN
    # miss_type - an integer specifying the desired type of missingness
    #   1 - missing completely at random
    #   2 - sensor drop-out: missing correlated in the rows
    #   3 - multi-rate: missing due to different sampling rates
    #   4 - censoring: missing due to particular values
    #   5 - patterned: similar to sensor where missing is correlated spatially
    #   but less random
    # level - percent of missing data, specified by a decimal
    # tol - tolerance to specify an acceptble deviation from the desired level
    # N.B. there are several factors that impact missingness that are not
    # variable in the below implementation. E.g. the maximum number of time
    # points for sensor drop-out missingness is 10. Hopefully these factors are
    # obvious and you can change them to suit your needs.
    #
    ###########################################################################
    #                                 Outputs                                 #
    ###########################################################################
    # X_miss - the data set with added NaNs to achieve the missing type and
    # level
    # miss_per - the final result for the level
    #
    #
    # For more information concerning types of missing data, please see
    # Rubin, D.B. Inference and missing data. Biometrika 1976, 63, 581-592.
    # Little, R.J.A. and Rubin, D.B. Statistical Analysis with Missing Data.
    # 2nd ed. John Wiley & Sons, 2014.
    #
    #
    # If you use this software, please cite
    # Severson, K.A., Molaro, M.C., and Braatz, R.D. Methods for applying
    # principal component analysis to process datasets wiht missing values.
    # Processes 2017, vol, pages.

    n = X.shape[0]
    p = X.shape[1]
    max_iter = 1000 #set maximum number of iterations
    iter = 0 #initialize iteration count
    #set missingness

    if miss_type == 1: #mcar
        X_miss = copy.deepcopy(X)
        base_miss = np.sum(np.isnan(X))/(n*p) #check if the dataset already has missing values
        while (abs(np.sum(np.isnan(X_miss))/(n*p) - level) > tol) & (iter < max_iter):
            X_miss = copy.deepcopy(X)
            wx = np.random.random((n, p)) < level - base_miss
            X_miss[wx] = np.nan
            iter = iter + 1
        miss_per = np.sum(np.isnan(X_miss))/(n*p)


    elif miss_type == 2: #sensor drop-out
        X_miss = copy.deepcopy(X)
        c_level = int(np.floor(p/4)) #number of measurements with missingness
        l_level = 10 #max length of time for missingness
        while (abs(np.sum(np.isnan(X_miss))/(n*p) - level) > tol) & (iter < max_iter):
            X_miss = copy.deepcopy(X)
            cx = np.random.randint(0, p, (c_level,1)) #choose measurements (cols) to have missing measurements
            rx = np.random.randint(0, n, (c_level,1)) #choose time index where missingness starts
            lx = np.random.randint(1, l_level+1, (c_level,1)) #choose length of time missingness occurs
            for i in range(c_level):
                if rx[i] + lx[i] > n:
                    X_miss[int(rx[i]):, cx[i]] = np.nan
                else:
                    X_miss[list(range(int(rx[i]), int(rx[i] + lx[i]))), cx[i]] = np.nan
            if np.sum(np.isnan(X_miss))/(n*p) - level > 0: #too much missing data
                c_level = c_level - 1 #decrease the number of measurements with missingness
            else:
                c_level = c_level + 1 #increase the number of measurements with missingness
            iter = iter + 1
        miss_per = np.sum(np.isnan(X_miss))/(n*p)

    elif miss_type == 3: #multi-rate
        # check if possible
        min_miss = 1/(2*p)
        if min_miss - level > tol:
            X_miss = copy.deepcopy(X)
            c_level = 1
            rx = 4
            cx = np.random.randint(0, p,(c_level,1))
            for i in range(c_level):
                X_miss[:,cx[i]] = np.nan
                X_miss[np.arange(0, n, rx[i]).tolist(),cx[i]] = X[np.arange(0, n, rx[i]).tolist(),cx[i]]
            print('Desired tolerance not possible, missing level set to %0.2f \n',min_miss)
        else:
            X_miss = copy.deepcopy(X)
            c_level = int(np.floor(p/3)) #number of measurements with missingess
            r_level = 5 #max level of subsampling
            while (abs(np.sum(np.isnan(X_miss))/(n*p) - level) > tol) & (iter < max_iter):
                X_miss = copy.deepcopy(X)
                cx = np.random.choice(range(0, p), c_level, replace=False)
                rx = np.random.choice(range(2, r_level+1), c_level)
                for i in range(c_level):
                    #X_miss[:,cx[i]] = np.nan
                    #X_miss[np.arange(0, n, rx[i]).tolist(),cx[i]] = X[np.arange(0, n, rx[i]).tolist(),cx[i]]
                    X_miss[np.arange(0, n, rx[i]).tolist(), cx[i]] = np.nan
                if np.sum(np.isnan(X_miss))/(n*p) - level > 0: #too much missing data
                    c_level = c_level - 1 #decrease number of measurements with missingness
                else:
                    c_level = c_level + 1 #increase number of measurements with missingness
                iter = iter + 1
        miss_per = np.sum(np.isnan(X_miss))/(n*p)

    elif miss_type == 4: #nmar
        X_miss = copy.deepcopy(X)
        min_c_level = int(np.floor(p*level)) + 2
        max_c_level = min(min_c_level + np.floor(p/4),p)
        c_level = np.random.randint(min_c_level,max_c_level) #number of measurements with missingness
        meas = np.random.choice(p, c_level, replace=False)
        std_start = 1.5 * np.nanstd(X[:,meas], axis=0) #initialize the threshold as 1.5 standard deviations
        updown = np.sign(-1 + 2.*np.random.rand(1,len(meas))).flatten()
        thres = np.nanmean(X[:,meas], axis=0) + np.multiply(std_start, updown) #randomize up and down thresholds
        while (abs(np.sum(np.isnan(X_miss))/(n*p) - level) > tol) & (iter < max_iter):
            X_miss = copy.deepcopy(X)
            for i in range(len(meas)):
                if updown[i] < 0:
                    X_miss[X[:,meas[i]] < thres[i],meas[i]] = np.nan
                else:
                    X_miss[X[:,meas[i]] > thres[i],meas[i]] = np.nan
            if np.sum(np.isnan(X_miss))/(n*p) - level > 0: #too much missing data
                thres = thres + 0.01 * np.multiply(std_start, updown)
            else:
                thres = thres - 0.05 * np.multiply(std_start, updown)
            iter = iter + 1

        miss_per = np.sum(np.isnan(X_miss))/(n*p)

    elif miss_type == 5: #mar
        X_miss = copy.deepcopy(X)
        r_level = 0.25
        c_level = 0.55
        while (abs(np.sum(np.isnan(X_miss))/(n*p) - level) > tol) & (iter < max_iter):
            X_miss = copy.deepcopy(X)
            rx = (np.random.random((n, 1)) < r_level).flatten()
            cx = (np.random.random((p, 1)) < c_level).flatten()
            X_miss[np.ix_(rx,cx)] = np.nan
            if np.sum(np.isnan(X_miss))/(n*p) - level > 0: #too much missing data
                r_level = r_level/2
                c_level = c_level/2
            else:
                r_level = r_level*1.1
                c_level = c_level*1.1

            iter = iter + 1

        miss_per = np.sum(np.isnan(X_miss))/(n*p)



    return X_miss, miss_per


