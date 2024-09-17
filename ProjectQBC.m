%% Second part of B
clear all
load('myVariables.mat', 'Rain_vector');
load 'proj23 (1).mat'

nvdi = ElGeneina.nvdi;

%Normalize data [-1,1]
normalized_nvdi = nvdi./255;
scaled_nvdi = 2 * normalized_nvdi - 1;
nvdi = scaled_nvdi;

%Split into training and validation data
nvdi_train = nvdi(1:450); %Training data
nvdi_val = nvdi(1:585); %Validation data
nvdi_train = iddata(nvdi_train);
nvdi_val = iddata(nvdi_val);
nvdi_train = nvdi_train.y;
nvdi_val = nvdi_val.y;
nvdi_test = nvdi;
nvdi_test = iddata(nvdi_test);
nvdi_test=nvdi_test.y;
%% Extract correct timeperiod and normalize rain
Rain = Rain_vector(end-648+1:end)';
nbroflags = 50;
%Split into training and validation data
Rain_train = Rain(1:450); %Training data
Rain_val = Rain(1:585); %Validation data
Rain_train = iddata(Rain_train);
Rain_val = iddata(Rain_val);
Rain_train = Rain_train.y;
Rain_val = Rain_val.y;
Rain_test = iddata(Rain);
Rain_test = Rain_test.y;
testlimit = 585;

%% ARMA(37,3) a1 a36 a37 c3
model_init = idpoly([1 zeros(1,37)],[],[1 0 0 0]);
model_init.Structure.a.Free = [1 1 0 0 zeros(1,32) 1 1];
model_init.Structure.c.Free = [1 0 0 1];
model_ar37ma3= pem(Rain_train,model_init);

res1 = resid(model_ar37ma3,Rain_train);
res_rain_ar37ma3 = res1.y;

figure,
plot(res_rain_ar37ma3)

figure,
plotACFnPACF(res_rain_ar37ma3,nbroflags,'Rain Model');


present(model_ar37ma3)
checkIfWhite(res_rain_ar37ma3)
save('myRainProcess.mat','model_ar37ma3')

%%
n = 450;
A3 = model_ar37ma3.a;
C3 = model_ar37ma3.c;
w_t = filter(A3,C3,Rain_train);
eps_t = filter(A3,C3,nvdi_train);
M = 100; stem(-M:M, crosscorr(w_t,eps_t,M));
title('Cross correlation function w_t and eps_t'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), '--');
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1), '--');
% BÖRJAR HÄR IMORGON glöm inte att ta bort all initialvärden
d=2;
r=0;
s=0;
% d=2;
% r=2;
% s=3;
A2 = [ones(1,r+1)]; %Does not have to be 1 find better est
B = [zeros(1,d) ones(1,s+1)]; %does not have to be 1 find better est, number of zeros in b gives the delay
Mi = idpoly(1, B, [], [], A2);
z = iddata(nvdi_train,Rain_train);
Mba2 = pem(z,Mi); 
present(Mba2);
etilde = resid(Mba2,z);

figure,
plotAcfPacfNormP(etilde.y(4:end),nbroflags);
%% Check if etilde and x is uncorralted which it is not, which is semi okay not perfectly gaussian
figure;

% M = 40; stem(-M:M, crosscorr(etilde.y(3:end),Rain_train(3:end),M));
M = 40; stem(-M:M, crosscorr(etilde.y(6:end),Rain_train(6:end),M));
title('Cross correlation function etilde and x'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), '--');
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1), '--');
hold off
%% Find Model for etilde
model_init = idpoly([1 zeros(1,36)], [], [1 zeros(1,36)]);
model_init.Structure.a.Free = [1 1 zeros(1,34) 1];
model_init.Structure.c.Free = [1 0 zeros(1,34) 1];
model_etilde = pem(etilde.y,model_init);
res_etilde = resid(etilde.y,model_etilde);
figure,
plotACFnPACF(res_etilde.y(2:end),nbroflags,'Model etilde');
checkIfWhite(res_etilde.y(2:end));
present(model_etilde)

%% Create MBoxJ using parameters found as initial parameters

A1 = model_etilde.A;
A2 = Mba2.F;
B = Mba2.B;
C = model_etilde.C;
Mi = idpoly(1, B,C,A1,A2);
z = iddata(nvdi_train,Rain_train);
Mi.Structure.c.Free=[1 zeros(1,35) 1];
Mi.Structure.d.Free=[1 1 zeros(1,34) 1];
MboxJ = pem(z,Mi);
present(MboxJ);
ehat=resid(MboxJ,z);
%%
plot(ehat.y(36:end));
figure;
plotACFnPACF(ehat.y(36:end),nbroflags,'Residual from Box Jenkins');
checkIfWhite(ehat.y(36:end))

%%
%check if ehat and Rain is uncorrelated
figure;
M = 40; stem(-M:M, crosscorr(ehat.y(36:end),Rain_train(36:end),M));
title('Cross correlation Residual and Rain'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), '--');
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1), '--');
hold off
% 1/50 outside confidence interval on positive axis so yes uncorrelated
%% Test MBoxJ on validation, very good except for the first 5 values of the validation set 
z = iddata(nvdi_val,Rain_val);
ehat = resid(MboxJ,z);

figure,
ehat = ehat.y(end-135:end);
plot(ehat);

figure,
plotACFnPACF(ehat,nbroflags,'Residual on validation');

checkIfWhite(ehat);

%% Kalman filter for MBoxJ
B = MboxJ.b;
D = MboxJ.d;
F = MboxJ.f;
C = MboxJ.c;
k=1;
y = nvdi_val;
x = Rain_val;
modelLim = 450;
k=1;

%% Predict the output using the found model.
[Fx, Gx] = polydiv( model_ar37ma3.c, model_ar37ma3.a, k ); %Predict the input with found model for MBoxJ
xhatk = filter(Gx, model_ar37ma3.c, x);

 
KA = conv(D,F);
KB = conv(D,B);
KC = conv(F,C);
[Fy, Gy] = polydiv(C,D,k);
[Fhh, Ghh] = polydiv(conv(Fy,KB),KC,k);
yhatk = filter(Fhh, 1, xhatk) + filter(Ghh,KC,x)+filter(Gy, KC, y);
ep_train = y(1:modelLim)-yhatk(1:modelLim);
eP    = y(modelLim:end)-yhatk(modelLim:end);    % Form the prediction residuals for the validation data.


plotAcfPacfNormP(eP,nbroflags); 
checkIfWhite(eP); %1-step prediction is white
normalized_var = var(eP)/var(nvdi_val(450:585))
MSE_val = sum(eP.*eP)/length(eP)
%%
save('myKA.mat','KA');
save('myKB.mat','KB');
save('myKC.mat','KC');
%%
figure,
plot(yhatk)
hold on
plot(y)
%%
plotAcfPacfNormP(ep_train,nbroflags);
checkIfWhite(ep_train);
%%
figure,
plot(y(modelLim:end));
hold on
plot(yhatk(modelLim:end));
%% Predict the output using the found model on the test data.
k=7;
[Fx, Gx] = polydiv( model_ar37ma3.c, model_ar37ma3.a, k ); %Predict the input with found model for MBoxJ
xhatk = filter(Gx, model_ar37ma3.c, Rain_test);
[Fy, Gy] = polydiv(C,D,k);
[Fhh, Ghh] = polydiv(conv(Fy,KB),KC,k);

yhatk = filter(Fhh, 1, xhatk) + filter(Ghh,KC,Rain_test)+filter(Gy, KC, nvdi_test);
eP_test    = nvdi_test(testlimit:end)-yhatk(testlimit:end);    % Form the prediction residuals for the validation data.

figure,
plotAcfPacfNormP(eP_test,nbroflags);

figure,
plot(yhatk(testlimit:end))
hold on
plot(nvdi_test(testlimit:end))
checkIfWhite(eP_test); %k-step prediction is white

normalized_var = var(eP_test)/var(nvdi_test(585:end))
MSE_val = sum(eP_test.*eP_test)/length(eP_test)


%% Now to the Kalman Filter
noPar = 2 + 3 + 1; %no states which are non zero
y = nvdi_val;
y = nvdi_test;
% x = Rain_val;
x = Rain_test;
N = length(Rain_test);
A = eye(noPar);
% N = length(Rain_val);
Rw = std(eP);
% Rw = 0.1;
Re = 1e-9*eye(noPar); % Tried different Re much better with lower
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
Kalgain=zeros(6,N);

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
%%
figure,
plot(yhatK(startInd:end-1),'b')
hold on
plot(y(startInd:end-1),'r')

%% 1-step prediction
figure,
plot(yt1(startInd:end))
hold on
plot(y(startInd:end))
%%
figure,
plot(yt1(modelLim:end));
hold on
plot(nvdi_val(modelLim:end))

%%
res_tot = y(startInd:end)-yt1(startInd:end);
checkIfWhite(res_tot)
%%
res_val = y(modelLim:end)-yt1(modelLim:end);

checkIfWhite(res_val);
normalized_var = var(res_val)/var(y(modelLim:end))
MSE_val = sum(res_val.*res_val)/length(res_val)
%% 1-step Testdata
figure,
plot(yt1(testlimit:end));
hold on
plot(y(testlimit:end))

res_test = y(testlimit:end)-yt1(testlimit:end);
checkIfWhite(res_test);
normalized_var = var(res_test)/var(y(testlimit:end))
MSE_val = sum(res_test.*res_test)/length(res_test)

%% 7-step prediction
figure,
plot(yt7(modelLim:end),'r')
hold on
plot(y(modelLim:end),'b')

figure,
plot(yt7(startInd:end),'r')
hold on
plot(y(startInd:end),'b')

res_tot = y(startInd:end-7)-yt7(startInd:end-7);
checkIfWhite(res_tot);

res_val = y(modelLim:end)-yt7(modelLim:end);
figure,
plotAcfPacfNormP(res_val,nbroflags);
checkIfWhite(res_val(5:end));
normalized_var = var(res_val)/var(y(modelLim:end))
MSE_val = sum(res_val.*res_val)/length(res_val)

%% 7-step prediction on testdata
figure,
plot(yt7(testlimit:end),'r')
hold on
plot(y(testlimit:end),'b')

res_test = y(testlimit:end)-yt7(testlimit:end);
checkIfWhite(res_test);
normalized_var = var(res_test)/var(y(testlimit:end))
MSE_val = sum(res_test.*res_test)/length(res_test)
%% Naive comparator
ynaive = nvdi_test(testlimit-36*k:end-36*k);
figure,
plot(ynaive)
hold on
plot(yt7(testlimit:end))
plot(nvdi_test(testlimit:end))


%% Looking at whole set
figure,
plot(ehat(startInd:end));

checkIfWhite(ehat(startInd:end-1)); %Outlier at nvdi_val(1-5) which totally rocks the test, happens in many tests
%% See which ones could be dismissed or timeindependent
for k=1:noPar
    figure,
    plot(xt(k,:))
end
%%
for k=1:noPar
    var(xt(k,:))
end
%%
plot(nvdi_test);
hold on
plot(Rain_test/200)