%% Second part of C removing parameters
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


%% ARMA(37,3) a1 a36 a37 c3
model_init = idpoly([1 zeros(1,37)],[],[1 0 0 0]);
model_init.Structure.a.Free = [1 1 zeros(1,34) 1 1];
model_init.Structure.c.Free = [1 0 0 1];
model_ar37ma3= pem(Rain_train,model_init);


%%
n = 450;
A3 = model_ar37ma3.a;
C3 = model_ar37ma3.c;
w_t = filter(A3,C3,Rain_train);
eps_t = filter(A3,C3,nvdi_train);

d=2;
r=0;
s=0;

A2 = [ones(1,r+1)]; %Does not have to be 1 find better est
B = [zeros(1,d) ones(1,s+1)]; %does not have to be 1 find better est, number of zeros in b gives the delay
Mi = idpoly(1, B, [], [], A2);
z = iddata(nvdi_train,Rain_train);
Mba2 = pem(z,Mi); 
present(Mba2);
etilde = resid(Mba2,z);


%%
model_init = idpoly([1 zeros(1,36)], [], [1 zeros(1,36)]);
model_init.Structure.a.Free = [1 1 zeros(1,34) 1];
model_init.Structure.c.Free = [1 0 zeros(1,34) 1];
model_etilde = pem(etilde.y,model_init);

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




%% Kalman filter for MBoxJ
B = MboxJ.b;
D = MboxJ.d;
F = MboxJ.f;
C = MboxJ.c;
k=1;
y = nvdi_val;
x = Rain_val;
modelLim = 450;
%k=7;

 
KA = conv(D,F);
KB = conv(D,B);
KC = conv(F,C);

%% Now to the Kalman Filter
noPar = 3; % 2 + 3 + 1; %no states which are non zero
y = nvdi_val;
N = length(Rain_val);
A = eye(noPar);
Rw = 0.1;
Re = 1e-9*eye(noPar); % Tried different Re much better with lower
Rx_t1=1e-4*eye(noPar);
ehat = zeros(N,1);
yhatK = zeros(N,1);
yt7 = zeros(N,1);
yt1 = zeros(N,1);
xStd  = zeros(noPar,N); 
xhat = zeros(N,1);


k =1;
k=7;
%We have delay of 2 so wont need to prdict x until 3-step prediction
startInd = 37;
xt= zeros(noPar,N); 
xt(:,startInd-1) = [-KA(2) -KA(37) KC(37)];

for t=startInd:(N-k)
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    Ct = [ y(t-1) y(t-36) ehat(t-36)];  
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



    
    Ct1 = [ y(t) y(t-35) ehat(t-35)];  
    yhat1 = Ct1*x_t1;

    if k>1
    Ct2 = [yhat1 y(t-34) ehat(t-34)];  
    yhat2 = Ct2*x_t1;

    Ct3 = [yhat2 y(t-33) ehat(t-33)];  
    yhat3 = Ct3*x_t1;

    Ct4 = [yhat3 y(t-32) ehat(t-32)];  
    yhat4 = Ct4*x_t1;
    
    Ct5 = [yhat4 y(t-31) ehat(t-31)];  
    yhat5 = Ct5*x_t1;

    Ct6 = [yhat5 y(t-30) ehat(t-30)];  
    yhat6 = Ct6*x_t1;

    Ct7 = [yhat6 y(t-29) ehat(t-29)];  
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
plot(yt1)
hold on
plot(y)
%%
res_tot = y(startInd:end-k)-yt1(startInd:end-k);
checkIfWhite(res_tot)
%%
res_val = y(modelLim:end-k)-yt1(modelLim:end-k);
checkIfWhite(res_val);

%% 7-step prediction
figure,
plot(yt7(modelLim:end-7),'r')
hold on
plot(y(modelLim:end-7), 'b')

res_tot = y(startInd:end)-yt7(startInd:end);
checkIfWhite(res_tot);

res_val = y(modelLim:end)-yt7(modelLim:end);
figure,
plotAcfPacfNormP(res_val,nbroflags);
checkIfWhite(res_val);
%% Looking at whole set
figure,
plot(ehat(startInd:end));

checkIfWhite(ehat(startInd:end-1));

%% See which ones could be dismissed or timeindependent
for k=1:noPar
    figure,
    plot(xt(k,:))
end