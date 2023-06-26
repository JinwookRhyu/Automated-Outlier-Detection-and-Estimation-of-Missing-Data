function [X_list, if_error, A_list, time_list] = fill_missing(Xin, if_normalize, A)
% A function that fills in the missing values using the following 10 methods
    % MI: Mean Imputation
    % ALS: Alternating Least Squares using pca command
    % Alternating: Alternating Least Squares implemented by Ilin and Raiko
    % SVDImpute: SVD Imputation implemented by Ilin and Raiko
    % PCADA: PCA-data augmentation implemented by Imtiaz and Shah
    % PPCA: Probabilistic PCA using ppca command
    % PPCA-M: Modified PPCA implemented by Severson et al
    % BPCA: Bayesian PCA implemented by Oba et al
    % SVT: Singular Value Thresholding implemented by Cai et al
    % ALM: Augmented Lagrange Multiplier implemented by Lin et al
% Inputs:
    % Xin: [n x d] matrix with missing values expressed as NaN
    % if_normalize: Boolean whether we normalize the data before running the code
    % A: Number of principal components with using preprocessed dataset (whether Mean_Impute, Interpolation, or Last-one)
% Outputs:
    % X_list: [n x d x 10] array with the missing values filled in
        % X_list(:,:,1) for MI, X_list(:,:,2) for ALS, ..., X_list(:,:,10) for ALM
    % if_error: [1 x 10] array which informs whether an error has occurred in each algorithm
    % A_list: [1 x 10] array of the number of PCs used in each algorithm
    % time_list: [1 x 10] array of the compuation time for each algorithm

% Loading files needed for Alternating, SVDImpute, BPCA, and SVT
rng(0)
warning('off')
addpath('.\BPCA')

%% Normalize input if if_normalize == true
% Normalize the data before running the code
if if_normalize == true
    [Xin, C, S] = normalize(Xin,1);
end

%% Apply imputation algorithms
% Apply 10 methods to fill-in missing values

X_list = zeros(size(Xin,1), size(Xin,2), 10);
A_list = zeros(1, 10);
time_list = zeros(1,10);

%First filter: Check whether codes were well-executed
if_error = zeros(1, 10);

disp('=================================================================')
disp('Applying imputation algorithms')
disp('=================================================================')
%% Mean imputation (MI)
try
    tic
    [X_list(:,:,1), A_list(1)] = MI(Xin);
    time_list(1) = toc;
catch
    % Set if_error to 1 if caught an error during the code implementation
    if_error(1) = 1;
end
disp('Mean Imputation (MI)...... done')

%% Alternating least squares as implemented by pca command (ALS)

a = A; % At first, set the number of PCs to A
for kk = 1:5
    try
        tic
        [X_list(:,:,2), A_list(2)] = ALS(Xin, a, 5);
        time_list(2) = toc;
    catch
        if_error(2) = 1;
    end
    if a == A_list(2) % If the determined 'a' is same as the input 'a',
        break
    else
        a = A_list(2); % Set input 'a' as the determined 'a' and re-run the code until 'a' converges
    end
end

disp('Alternating Least Squares using pca command (ALS)...... done')

%% Alternating least squares implemented by Ilin and Raiko (Alternating)
a = A;
for kk = 1:5   
    try
        tic
        [X_list(:,:,3), A_list(3)] = Alternating(Xin, a);
        time_list(3) = toc;
    catch
        if_error(3) = 1;
    end
    if a == A_list(3)
        break
    else
        a = A_list(3);
    end
end

disp('Alternating Least Squares (Alternating)...... done')

%% SVDImpute implemented by Ilin and Raiko (SVDImpute)
a = A;
for kk = 1:5
    try
        tic
        [X_list(:,:,4), A_list(4)] = SVDImpute(Xin, a);
        time_list(4) = toc;
    catch
        if_error(4) = 1;
    end
    if a == A_list(4)
        break
    else
        a = A_list(4);
    end
end

disp('SVD Imputation (SVDImpute)...... done')


%% PCA-data augmentation implemented by Imtiaz and Shah (PCADA)
a = A;
for kk = 1:5  
    try
        tic
        [X_list(:,:,5), A_list(5)] = PCADA(Xin, a, 50);
        time_list(5) = toc;
    catch
        if_error(5) = 1;
    end
    if a == A_list(5)
        break
    else
        a = A_list(5);
    end
end

disp('PCA-data augmentation (PCADA)...... done')

%% MATLAB built-in PPCA command (PPCA)
a = A;
for kk = 1:5  
    try
        tic
        [X_list(:,:,6), A_list(6)] = PPCA(Xin, a);
        time_list(6) = toc;
    catch
        if_error(6) = 1;
    end
    if a == A_list(6)
        break
    else
        a = A_list(6);
    end
end

disp('Probabilistic PCA (PPCA)...... done')

%% PPCA-M implemented by the authos (PPCA-M)
a = A;
for kk = 1:5   
    try
        tic
        [X_list(:,:,7), A_list(7)] = PPCAM(Xin, a);
        time_list(7) = toc;
    catch
        if_error(7) = 1;
    end
    if a == A_list(7)
        break
    else
        a = A_list(7);
    end
end

disp('Probabilistic PCA for missing data (PPCA-M)...... done')

%% Bayesian PCA implemented by Oba et al (BPCA)
a = A;
for kk = 1:5
    try
        tic
        [X_list(:,:,8), A_list(8)] = BPCA(Xin, a);
        time_list(8) = toc;
        if sum(sum(abs(X_list(:,:,8) - 999) < 0.01)) > 0
            if_error(8) = 1;
        end
    catch
        if_error(8) = 1;
    end
    if a == A_list(8)
        break
    else
        a = A_list(8);
    end
end

disp('Bayesian PCA (BPCA)...... done')

%% Singular value thresholding implemented by Cai et al (SVT)

try
    tic
    [X_list(:,:,9), A_list(9)] = SVT(Xin);
    time_list(9) = toc;
catch
    if_error(9) = 1;
end
disp('Singular Value Thresholding (SVT)...... done')

%% Augmented Lagrange Multiplier by Lin et al (ALM)

try
    tic
    [X_list(:,:,10), A_list(10)] = ALM(Xin);
    time_list(10) = toc;
catch
    if_error(10) = 1;
end
disp('Augmented Lagrange Multiplier (ALM)...... done')

%% Transform back if the data was normalized before running the code
if if_normalize == true
    X_list = X_list .* S + C;
end

end

%% Functions
function [Xout, Aout] = MI(Xin)
    
    X = Xin;
    M = isnan(X);
    X_mean = mean(X, 1, 'omitnan');
    Xout = X;
    for j = 1:size(X,2)
        Xout(isnan(X(:,j)), j) = X_mean(j);
    end
    
    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);
    
end

function [Xout, Aout] = ALS(Xin, Ain, numiterals)
    X = Xin;
    M = isnan(X);
    
    CONT_M_list = zeros(numiterals,1);
    X_ALS_list = zeros(size(Xin,1), size(Xin,2), numiterals);
  
    for ii = 1:numiterals
        warning('off')
        [coeff1,score1,latent1,~,~,mu1] = pca(X,'algorithm','als','NumComponents',Ain, 'Centered', false);
        Xout = score1 * coeff1' + repmat(mu1,size(X,1),1);
        
        X_ALS_list(:,:,ii) = Xout;
        X = Xout;
        [n,d] = size(X);
        CONT = zeros(n,d);
        
        for k = 1:n
            t = X(k,:) * coeff1;
            cont = zeros(Ain, d);
            for i = 1:Ain
                for j = 1:d
                    cont(i,j) = t(i) * X(k,j) * coeff1(j,i) / latent1(i);
                end
            end
            cont = cont .* (cont > 0);
            CONT(k,:) = sum(cont);
        end

        CONT_M_list(ii) = max(max(CONT.*M));
    end
    
    [~, index] = min(CONT_M_list);
    Xout = X_ALS_list(:,:,index);
    
    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);
    
end

function [Xout, Aout] = Alternating(Xin, Ain)
    
    SSE_old = 1;

    X = Xin;    
    X0 = X;
    X0(isnan(X)) = 0;
    M = isnan(X);
    X = X0;

    [P, T] = pca(X0, 'NumComponents', Ain);
    mu = zeros(size(X0,2), 1);
    T_dev = zeros(size(X0,1), Ain, size(X0,2));
    P_dev = zeros(size(X0,2), Ain, size(X0,1));
    mu_dev = zeros(size(X0,2), size(X0,1));
    
    for j = 1:size(X0,2)
        O = find(M(:,j) == 0);
        T_dev(O, :, j) = T(O, :);
    end
    for i = 1:size(X0,1)
        O = find(M(i,:) == 0);
        P_dev(O, :, i) = P(O, :);
    end

    for ii = 1:1000
        
        for i = 1:size(X0,1)
            T(i,:) = (inv(P_dev(:,:,i)' * P_dev(:,:,i)) * P_dev(:,:,i)' * (X0(i,:)' - mu_dev(:,i)))';
        end
        T_dev = zeros(size(X0,1), Ain, size(X0,2));
        for j = 1:size(X0,2)
            O = find(M(:,j) == 0);
            T_dev(O, :, j) = T(O, :);
        end
        
        mu = zeros(size(X0,2), 1);
        mu_dev = zeros(size(X0,2), size(X0,1));
        for j = 1:size(X0,2)
            O = find(M(:,j) == 0);
            mu(j) = 1 / length(O) * sum((X(:, j) - T * P(j, :)') .* (1 - M(:, j)));
            mu_dev(j, O) = mu(j);
        end

        for j=1:size(X0,2)
            P(j, :) = (X0(:, j) - mu(j))' * (T_dev(:, :, j) * inv(T_dev(:,:,j)' * T_dev(:,:,j)));
        end
        P_dev = zeros(size(X0,2), Ain, size(X0,1));
        for i = 1:size(X0,1)
            O = find(M(i,:) == 0);
            P_dev(O, :, i) = P(O, :);
        end

        SSE = sum(sum(((X0 - T * P') .* (1-M)).^2));
        rel_error = abs((SSE - SSE_old) / SSE_old);
        SSE_old = SSE;
        
        if rel_error < 10^(-6)
            break
        elseif SSE < 10^(-10)
            break
        end
    end

    Xout = T * P';
  
    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);
   
end

function [Xout, Aout] = SVDImpute(Xin, Ain)
    
    X = Xin;    
    X0 = X;
    X0(isnan(X)) = 0;
    M = isnan(X);
    X_rec = X0;
    
    SSE_obs_list = zeros(1000, 1);
    SSE_obs_old = 0;
    for ii = 1:1000
        S = 1 / (size(X_rec, 1) - 1) * (X_rec' * X_rec);
        [V, ~, ~] = svd(S);
        P = V(:, 1:Ain);
        T = X_rec * P;
        SSE_obs_list(ii) = sum(sum(((X_rec - T * P').^2) .* (1 - M))) / sum(sum(1 - M));
        SSE_obs_new = SSE_obs_list(ii);
        X_rec = X_rec .* (1 - M) + (T * P') .* M;

        if abs((SSE_obs_new - SSE_obs_old) / SSE_obs_new) < 10^(-6)
            break
        end
        SSE_obs_old = SSE_obs_new;
    end

    Xout = X_rec;

    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);
   
end

function [Xout, Aout] = PCADA(Xin, Ain, K)
    
    X = Xin;    
    X0 = X;
    X0(isnan(X)) = 0;
    M = isnan(X);
    SSE_old = 10;
    
    X_mis = repmat(mean(X0,1), size(X0,1), 1);

    X_hat = X0 .* (1-M) + X_mis .* M;

    Cov = (X_hat' * X_hat) / size(X0,1);
    [U,D] = eig(Cov);
    if ~issorted(diag(D), 'descend')
        [U,D] = eig(Cov);
        [D,I] = sort(diag(D), 'descend');
        U = U(:, I);
        D = diag(D);
    end
    
    P_hat = U(:,1:Ain);
    X_hat_K = repmat(X_hat, 1, 1, K);
    P_hat_K = repmat(P_hat, 1, 1, K);

    for ii = 1:1000
        
        X_nf = X_hat * P_hat * P_hat';
        r = X0 - X_nf;

        for k = 1:K
            X_hat_K(:,:,k) = X_hat_K(:,:,k) .* (1-M) + X_nf .* M;
            for j = 1:size(X0,2)
                ind = find(M(:,j));
                O = setdiff(1:size(X0,1), ind);
                X_hat_K(ind,j,k) = X_hat_K(ind,j,k) + r(O(randi(length(O), length(ind), 1)), j);
            end

            Cov = (X_hat_K(:,:,k)' * X_hat_K(:,:,k)) / size(X0,1);
            [U,D] = eig(Cov);
            if ~issorted(diag(D), 'descend')
                [U,D] = eig(Cov);
                [D,I] = sort(diag(D), 'descend');
                U = U(:, I);
            end

            P_hat_K(:,:,k) = U(:,1:Ain);  
        end

        P_hat = mean(P_hat_K,3);
        X_hat = mean(X_hat_K,3);
        SSE = sum(sum(((X_nf - X0) .* (1-M)).^2));
        rel_error = abs((SSE - SSE_old) / SSE_old);
        SSE_old = SSE;
        
        if rel_error < 10^(-6)
            break
        elseif SSE < 10^(-10)
            break
        end
    end

    Xout = X_hat;    
   
    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);

end

function [Xout, Aout] = PPCA(Xin, Ain)
    
    X = Xin;    
    M = isnan(X);
           
    [~,~,~,~,~,St] = ppca(X,Ain);
    Xout = St.Recon;

    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);
end

function [Xout, Aout] = PPCAM(Xin, Ain)
    
    X = Xin;
    X0 = X;
    X0(isnan(X)) = 0;
    M = isnan(X);
    O = 1 - M;
    
    P = pca(X0, 'NumComponents', Ain);
    X = X0;
    mu = mean(X,1);
    sigma = var(X(:));

    W = zeros(Ain,  Ain,  size(X0,1));

    for i = 1:size(X0,1)
        ind = find(O(i,:));
        W(:,:,i) = W(:,:,i) + P(ind,:)' * P(ind,:) + sigma * eye(Ain);
    end

    for ii = 1:1000

        sigma_old = sigma;

        % Upadate ti based on Eq.(29)
        T = zeros(size(X0,1), Ain);

        for i = 1:size(X0,1)
            ind = find(O(i,:));
            rest = P(ind,:)' * (X(i, ind) - mu(ind))';
            T(i,:) = (W(:,:,i) \ rest)';
        end

        % Update xij based on Eq.(30)
        for i = 1:size(X0,1)
            ind = find(M(i,:));
            X(i, ind) = T(i,:) * P(ind, :)' + mu(ind);
        end

        % Update titiT based on Eq.(31)
        titiT = zeros(Ain, Ain, size(X0,1));
        for i = 1:size(X0,1)
            titiT(:,:,i) = sigma * inv(W(:,:,i)) + T(i,:)' * T(i,:);
        end

        % Update xixiT based on Eq.(32)
        xixiT = zeros(size(X0,2), size(X0,2), size(X0,1));
        for i = 1:size(X0,1)
            xixiT(:,:,i) = (X(i,:)' * X(i,:)) .* (1 - M(i, :)' * M(i, :)) + (sigma * (eye(size(X0,2)) + (P / W(:,:,i)) * P') + X(i,:)' * X(i,:)) .* (M(i, :)' * M(i, :));
        end

        % Update xitiT based on Eq.(33)
        xitiT = zeros(size(X0,2), Ain,  size(X0,1));
        for i = 1:size(X0,1)
            xitiT(:, :, i) = M(i,:) .* sigma * (P / W(:,:,i)) + X(i,:)' * T(i,:);
        end

        % Update W, mu, P, sigma based on Eqs.(34)-(37)

        mu = 1 / size(X0,1) * sum((X - T * P'), 1); % Eq.(34)

        sumP1 = zeros(size(X0,2), Ain);
        for i = 1:size(X0,1)
            sumP1 = sumP1 + xitiT(:,:,i) - mu' * T(i,:);
        end
        sumP2 = sum(titiT,3);
        P = sumP1 / sumP2;

        sumsigma = 0;
        for i = 1:size(X0,1)
            sumsigma = sumsigma + trace(xixiT(:,:,i) - 2*(xitiT(:,:,i) * P') - 2 * mu' * X(i,:) + 2 * mu' * (T(i,:) * P') + P * titiT(:,:,i) * P' + mu' * mu);
        end
        sigma = sumsigma / size(X0,1) / size(X0,2);

        W = zeros(Ain,  Ain,  size(X0,1));
        for i = 1:size(X0,1)
            ind = find(O(i,:));
            W(:,:,i) = P(ind, :)' * P(ind, :) + sigma * eye(Ain);
        end

        %SSE
        rel_error = abs((sigma-sigma_old) / sigma_old);

        if rel_error < 10^(-6)
            break
        end 
    end

    Xout = X;
    
    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);
end

function [Xout, Aout] = BPCA(Xin, Ain)
    
    X = Xin;
    M = isnan(X);
    X(isnan(X)) = 999.00;
    
    Xout = BPCAfill(X, Ain, 1000);  
    
    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);
end

function [Xout, Aout] = SVT(Xin)
    
    X = Xin;
    X0 = X;
    X0(isnan(X)) = 0;
    M = isnan(X);
    
    tau = 5 * size(X0,1);
    tol = 10^(-4);
    delta = 1.2 * size(X0,1) * size(X0,2) / (size(X0,1) * size(X0,2) - sum(sum(M)));
    k0 = ceil(tau / delta / norm(X0, 'fro'));
    Z = k0 * delta * X0;

    for ii = 1:1000
        [U, S, Vh] = svd(Z);
        S = diag(S); Vh = Vh';
        r = sum(S > tau);
        U = U(:, 1:r);
        Vh = Vh(1:r, :);
        S = S(1:r);
        Amat = U * diag(S - tau) * Vh;
        dnorm = norm((Amat - X0) .* (1-M), 'fro') / norm(X0, 'fro');
        if dnorm < tol
            break
        end
        Z = Z + delta * (X0 - Amat) .* (1 - M);
    end
    
    Xout = Amat;
    
    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);
end

function [Xout, Aout] = ALM(Xin)
    
    X = Xin;
    M = isnan(X);
    X(isnan(X)) = 0;
    
    D = X;
    Y_old = zeros(size(X));
    E_old = zeros(size(X));
    mu_old = 1 / norm(D, 'fro');
    rho = 1.2172 + 1.8588 * (1 - sum(sum(M)) / size(X,1) / size(X,2));

    for ii = 1:1000
        [U, S, Vh] = svd(D - E_old + 1 / mu_old * Y_old);
        Vh = Vh';
        S = diag(S);
        U = U(:, 1:length(S));
        Amat = U * ((diag(S) - 1 / mu_old) .* (diag(S) > 1 / mu_old) + (diag(S) + 1 / mu_old) .* (diag(S) < -1 / mu_old)) * Vh;
        
        E_new = M .* (D - Amat + 1 / mu_old * Y_old);
        Y_new = Y_old + mu_old * (D - Amat - E_new);
    
        if min(mu_old, sqrt(mu_old)) * norm(E_new - E_old, 'fro') / norm(D, 'fro') < 10^(-6)
            mu_new = mu_old * rho;
        else
            mu_new = mu_old;
        end
    
        ratio1 = norm(D - Amat - E_new, 'fro') / norm(D, 'fro');
        ratio2 = min(mu_new, sqrt(mu_new)) * norm(E_new - E_old, 'fro') / norm(D, 'fro');
    
        if (ratio1 < 10^(-7)) && (ratio2 < 10^(-6))
            break
        end

        mu_old = mu_new;
        E_old = E_new;
        Y_old = Y_new;
    end

    Xout = Amat;

    X0 = Xin;
    X0(isnan(Xin)) = 0;
    Xout = X0 .* (1-M) + Xout .* M;
    
    % Model selection based on cross-validation
    RMSECV = cross_validate_pca(Xout, size(Xin, 2), 'G_obs', 7);
    [~, Aout] = min(RMSECV);
end
