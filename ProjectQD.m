%% Project QD
clear all
load('myVariables2.mat', 'Rain_vector_Kassala');
load('myNVDIprocess.mat','model_ar1ma37')
load 'proj23 (1).mat'
load('myRainProcess.mat','model_ar37ma3')
load('myKA.mat','KA');
load('myKB.mat','KB');
load('myKC.mat','KC');

nvdi = Kassala.nvdi;

%Normalize data [-1,1]
normalized_nvdi = nvdi./255;
scaled_nvdi = 2 * normalized_nvdi - 1;
nvdi = scaled_nvdi;

nvdi_test = iddata(nvdi);
nvdi_test=nvdi_test.y;
%% Extract correct timeperiod and normalize rain
Rain = Rain_vector_Kassala(end-648+1:end)';
nbroflags = 50;

Rain_test = iddata(Rain);
Rain_test = Rain_test.y;
testlimit = 585;
%%
k=7;
AS = [1 zeros(1,35) -1];
Aconv = conv(AS,model_ar1ma37.a); %get correct a without seasonality
[~, Gx] = polydiv( model_ar1ma37.c, Aconv, k ); 
yhatk = filter(Gx, model_ar1ma37.c, nvdi_test);

figure,
plot(yhatk)
hold on
plot(nvdi_test)

res = nvdi_test - yhatk;
figure,
plotACFnPACF(res,40,'1 Step Polydiv without input Kassala')
checkIfWhite(res)
normalized_var = var(res)/var(nvdi_test)
MSE_val = sum(res.*res)/length(res)


%% Now to the Kalman Filter
noPar = 2 + 3 + 1; %no states which are non zero
y = nvdi_test;
x = Rain_test;
N = length(Rain_test);
A = eye(noPar);

Rw = 0.0524; %Std of residual
% Rw = 0.1;
Re = 1e-9*eye(noPar); %Same as model in ElGeneina
Rx_t1=1e-2*eye(noPar);
ehat = zeros(N,1);
yhatK = zeros(N,1);
yt7 = zeros(N,1);
yt1 = zeros(N,1);
xStd  = zeros(noPar,N); 
xhat = zeros(N,1);
xhat1 = zeros(N,1);
xhat2 = zeros(N,1);
xhat3 = zeros(N,1);
xhat4 = zeros(N,1);
xhat5 = zeros(N,1);


k =1;
k=7;
%We have delay of 2 so wont need to prdict x until 3-step prediction
startInd = 39;
xt= zeros(noPar,N); 
xt(:,startInd-1) = [-KA(2) -KA(37) KB(3) KB(4) KB(39) KC(37)];

for t=startInd:(N-k)
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    Ct = [ y(t-1) y(t-36) x(t-2) x(t-3) x(t-38) ehat(t-36)];  
    yhatK(t) = Ct*x_t1;
    


    % Update the parameter estimates.
    Ry = Ct*Rx_t1*Ct' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*Ct'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    ehat(t) = y(t)-yhatK(t);                    % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*(ehat(t));            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

     % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

     % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );             % This is one std for each of the parameters for the one-step prediction.
    x_t1 = A*xt(:,t);                         % x_{t|t-1} = A x_{t-1|t-1}



    
    Ct1 = [ y(t) y(t-35) x(t-1) x(t-2) x(t-37) ehat(t-35)];  
    yhat1 = Ct1*x_t1;

    if k>1
    Ct2 = [yhat1 y(t-34) x(t) x(t-1) x(t-36) ehat(t-34)];  
    yhat2 = Ct2*x_t1;

    [~, Gx] = polydiv( model_ar37ma3.c, model_ar37ma3.a, 1 ); 
    xhat1 = filter(Gx, model_ar37ma3.c, x(1:t));
    if xhat1(t) < 0
        xhat1(t) = 0;
    end
    

    Ct3 = [yhat2 y(t-33) xhat1(t) x(t) x(t-35) ehat(t-33)];  
    yhat3 = Ct3*x_t1;

    [~, Gx] = polydiv( model_ar37ma3.c, model_ar37ma3.a, 2 ); %Predicting x should i do this for each k?
    xhat2 = filter(Gx, model_ar37ma3.c, x(1:t));
    
    if xhat2(t) < 0
        xhat2(t) = 0;
    end

    Ct4 = [yhat3 y(t-32) xhat2(t) xhat1(t) x(t-34) ehat(t-32)];  
    yhat4 = Ct4*x_t1;
    
    [~, Gx] = polydiv( model_ar37ma3.c, model_ar37ma3.a, 3); %Predicting x should i do this for each k?
    xhat3 = filter(Gx, model_ar37ma3.c, x(1:t));
    
    if xhat3(t) < 0
        xhat3(t) = 0;
    end
  
    Ct5 = [yhat4 y(t-31) xhat3(t) xhat2(t) x(t-33) ehat(t-31)];  
    yhat5 = Ct5*x_t1;

    [~, Gx] = polydiv( model_ar37ma3.c, model_ar37ma3.a, 4 ); %Predicting x should i do this for each k?
    xhat4 = filter(Gx, model_ar37ma3.c, x(1:t));

    if xhat4(t) < 0
        xhat4(t) = 0;
    end
   

    Ct6 = [yhat5 y(t-30) xhat4(t) xhat3(t) x(t-32) ehat(t-30)];  
    yhat6 = Ct6*x_t1;

    [~, Gx] = polydiv( model_ar37ma3.c, model_ar37ma3.a, 5 ); %Predicting x should i do this for each k?
    xhat5 = filter(Gx, model_ar37ma3.c, x(1:t));
    if xhat5(t) < 0
        xhat5(t) = 0;
    end
    Ct7 = [yhat6 y(t-29) xhat5(t) xhat4(t) x(t-31) ehat(t-29)];  
    yhat7 = Ct7*x_t1;

    yt7(t+7) = yhat7;
    end

    yt1(t+1) = yhat1;

end
%% Kassala 7-step
figure,
plot(yt7(startInd:end))
hold on
plot(y(startInd:end))



res = y(startInd:end)-yt7(startInd:end);
checkIfWhite(res) %for fun
normalized_var = var(res)/var(nvdi_test(startInd:end))
MSE_val = sum(res.*res)/length(res)

figure,
plot(res)

%% Kassala 1-step Testdata
figure,
plot(yt1(startInd:end));
hold on
plot(y(startInd:end))

res_test = y(startInd:end)-yt1(startInd:end);
checkIfWhite(res_test);
normalized_var = var(res_test)/var(y(startInd:end))
MSE_val = sum(res_test.*res_test)/length(res_test)
%%
for k=1:noPar
    figure,
    plot(xt(k,:))
end
