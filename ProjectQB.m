%Project QB
clear all
load 'proj23 (1).mat'

nvdi = ElGeneina.nvdi;


%Normalize data [-1,1]
normalized_nvdi = nvdi./255;
scaled_nvdi = 2 * normalized_nvdi - 1;
nvdi = scaled_nvdi;

AS = [1 zeros(1,35) -1];
nvdi = filter(AS,1,nvdi);



%Split into training and validation data
nvdi_train = nvdi(1:450); %Training data
nvdi_val = nvdi(1:585); %Validation data
nvdi_train = iddata(nvdi_train);
nvdi_val = iddata(nvdi_val);
nvdi_train = nvdi_train.y;
nvdi_val = nvdi_val.y;
nvdi_test = scaled_nvdi;
testlimit = 585;
%plot(ndvi_train)
nbroflags = 50;
figure,
plotAcfPacfNormP(nvdi_train,nbroflags);
figure,
plotAcfPacfNormP(nvdi_val,nbroflags);

%% ARMA(1,37) a1 c1 c36 c37
S = 36;
model_init=idpoly([1 0], [], [1 zeros(1,37)]);
model_init.Structure.c.Free = [1 1 zeros(1,34) 1 1];
model_ar1ma37 = pem(nvdi_train,model_init);
res_train = resid(nvdi_train,model_ar1ma37);
res_train = res_train.y(S:end);
figure,
plotACFnPACF(res_train,nbroflags,'NVDI');
checkIfWhite(res_train)
save('myNVDIprocess.mat', 'model_ar1ma37')
%% Using polydiv to predict without input
k =1;
Aconv = conv(AS,model_ar1ma37.a); %get correct a without seasonality
[~, Gx] = polydiv( model_ar1ma37.c, Aconv, k ); 
yhatk = filter(Gx, model_ar1ma37.c, scaled_nvdi(1:585));

figure,
plot(scaled_nvdi(450:585))
hold on
plot(yhatk(450:585))

res = scaled_nvdi(450:585) - yhatk(450:585);
figure,
plotACFnPACF(res,nbroflags,'1 Step Polydiv without input')
checkIfWhite(res)
normalized_var = var(res)/var(scaled_nvdi(450:585))
MSE_val = sum(res.*res)/length(res)


%% Using polydiv to predict without input on test data
k =7;
Aconv = conv(AS,model_ar1ma37.a); %get correct a without seasonality
[~, Gx] = polydiv( model_ar1ma37.c, Aconv, k ); 
yhatk = filter(Gx, model_ar1ma37.c, nvdi_test);

figure,
plot(yhatk(testlimit:end))
hold on
plot(nvdi_test(testlimit:end))

res = nvdi_test(testlimit:end) - yhatk(testlimit:end);
figure,
plotACFnPACF(res,40,'1 Step Polydiv without input')
checkIfWhite(res)
normalized_var = var(res)/var(nvdi_test(testlimit:end))
MSE_val = sum(res.*res)/length(res)
%% Plotting realization on testdata with best predictor and naive predictor k-step
ynaive = nvdi_test(testlimit-36*k:end-36*k);
figure,
plot(ynaive)
hold on
plot(yhatk(testlimit:end))
plot(nvdi_test(testlimit:end))

res = nvdi(testlimit:end)-ynaive;
MSE_naive = sum(res.*res)/length(res)

%% Predicting the vegetation without input data but with time variant (Remember to check on validation (only res on last samples))
d=6;
k=1;
A = eye(d);
Re = 1e-6*diag(d);
Rw = 0.1;
%y = scaled_nvdi(1:450); %train
%N = length(nvdi_train); %train
y = scaled_nvdi(1:585); %val
y = scaled_nvdi;
N = length(y); %val

%Set some initial values
Rxx_1 = 10*eye(6);   %Initial state variance
        %Obesrvation values = 2*N zeros
 
Aconv = conv(AS,model_ar1ma37.a); %get correct a without seasonality
%%
% Vectors to store values in
Xsave = zeros(d,N-k);
Xsave(:,37) = [Aconv(2) Aconv(37) Aconv(38) model_ar1ma37.c(2) model_ar1ma37.c(37) model_ar1ma37.c(38)]; %Stored states
ehat = zeros(1,N);  %Prediction residual
yt1 = zeros(1,N-k);   %One step prediction

yhat = zeros(N-k,1);


% The filter use data up to time t-1 to predict value at t,
% then update using the predictionerror. Why do we start
% from t=3? Why stop at N-2?

for t=38:N-k
    Ct = [-y(t-1) -y(t-36) -y(t-37) ehat(t-1) ehat(t-36) ehat(t-37)];     %C_{t|t-1}
    yhat(t) = Ct*Xsave(:,t-1);    %y_{t|t-1}
    ehat(t) = y(t)-yhat(t);    %e_t = y_t -y_{t|t-1}

    %Update     
    Ryy = Ct*Rxx_1*Ct'+Rw;        %R^{yy}_{t|t-1}
    Kt = Rxx_1*Ct'/(Ryy);         %K_t
    xt_t = Xsave(:,t-1)+Kt*(ehat(t));       %x_{t|t}
    Rxx = Rxx_1-Kt*Ryy*Kt';        %R{xx}_{t|t}

    %Predict the next state
    Xsave(:,t) = A*xt_t;      %x_{t+1|t}
    Rxx_1 = A*Rxx*A'+ Re;      %R^{xx}_{t+1|t}

    %Form 1-step precition
    Ct1 = [-y(t) -y(t-35) -y(t-36) ehat(t) ehat(t-35) ehat(t-36)];    %C_{t+1|t}
    ytk= Ct1*xt_t;
    %y_{t+1|t}=C_{t+1|t} x_{t|t}
    for k0=2:k
         Ctk = [-ytk -y(t-35+k0) -y(t-36+k0) ehat(t+k0) ehat(t-35+k0) ehat(t-36+k0)];
         ytk = Ctk*xt_t;     %y_{t+1|t}=C_{t+1|t} x_{t|t}
    end

     yt1(t+k) = ytk; 

end


%%
figure,
plot(Xsave(1,:));
hold on
yline(Aconv(2))
plot(Xsave(2,:));
yline(Aconv(37))
plot(Xsave(3,:));
yline(Aconv(38))
plot(Xsave(4,:))
yline(model_ar1ma37.c(2))
plot(Xsave(5,:))
yline(model_ar1ma37.c(37))
plot(Xsave(6,:))
yline(model_ar1ma37.c(38))
hold off

%%
figure,
plot(yt1(38:end))
hold on 
plot(y(38:end))

%%
figure,
plot(yt1(450:end))
hold on
plot(y(450:end))
%% Test results
figure,
plot(yt1(testlimit:end))
hold on 
plot(y(testlimit:end))
res_test = y(testlimit:end)-yt1(testlimit:end)';
checkIfWhite(res_test)
normalized_var = var(res_test)/var(nvdi_test(testlimit:end))
MSE_val = sum(res_test.*res_test)/length(res_test)


%% Check model on test data
resid1step = y(38:end) -yt1(38:end)';
plot(resid1step)
figure,
plotAcfPacfNormP(resid1step,nbroflags)
checkIfWhite(resid1step)

%% Check model on validation data
resid1step = y(end-135:end) -yt1(end-135:end)';
plot(resid1step)
figure,
plotAcfPacfNormP(resid1step,nbroflags)
checkIfWhite(resid1step)

normalized_var = var(resid1step)/var(scaled_nvdi(450:585))
MSE_val = sum(resid1step.*resid1step)/length(resid1step)