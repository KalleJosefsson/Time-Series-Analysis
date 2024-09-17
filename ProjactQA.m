% Project Q: A)

load 'proj23 (1).mat'
nbroflags = 50;
figure,
plot(ElGeneina.rain)
% 
% plot(Kassala.rain)
EG_Rain = log(ElGeneina.rain_org+1);
figure,
plot(EG_Rain)
%%
% figure
% plot(ElGeneina.rain_org)
% plotAcfPacfNormP(ElGeneina.rain_org,nbroflags);

%Extract A for convution
datays_forA = iddata(EG_Rain);
model_init = idpoly([1 0], [], []);
model_forA = pem(datays_forA,model_init);


S =12;
A12 = [1 zeros(1,S-1) -1];
y_s = filter(A12,1,EG_Rain);
y_s = y_s(length(A12):end);
y_s = y_s+5;
datays_rain = iddata(y_s);


model_init = idpoly([1 0], [], []);
model_rain_ar1x12 = armax(datays_rain,model_init);
res = resid(datays_rain,model_rain_ar1x12);
res = res.y(2:end);
figure,
plotAcfPacfNormP(res,nbroflags);
%%
% Kalman Filter Implementation
% y = ElGeneina.rain_org;% Observed data
y = Kassala.rain_org;
N = length(y);            % Number of observations

% State space model setup
Acoff = -model_rain_ar1x12.a(2); % 
A = [0 0 Acoff^2; 0 0 Acoff; 0 0 1];               % State transition matrix
% A = [Acoff^3 0 0; 0 Acoff^3 0; 0 0 Acoff^3];
%A = eye(3);

Re = 0.1*eye(3); % State covariance matrix
%Rw = var(res);             % Observation variance
Rw =0.1;

% Initial values
Rxx_1 = 10 * eye(3);      % Initial state variance/HIgh value i dont trust


% Storage vectors
Xsave = zeros(3, N);      % Stored states
ehat = zeros(1, N);       % Prediction residual
yhat = zeros(1, N);       % Predicted observation
xt_t1 = [0 0 0]';

for t=4:N
    Ct = [1 1 1];     %C_{t|t-1} xt + xt-1 +xt-2
    yhat(t) = Ct*xt_t1;    %y_{t|t-1}
    ehat(t) = y(t)-yhat(t);    %e_t = y_t -y_{t|t-1}

    %Update     
    Ryy = Ct*Rxx_1*Ct'+Rw;        %R^{yy}_{t|t-1}
    Kt = Rxx_1*Ct'/(Ryy);         %K_t
    xt_t = xt_t1+Kt*ehat(t);       %x_{t|t}
    Rxx = Rxx_1-Kt*Ryy*Kt';        %R{xx}_{t|t}

    %Predict the next state
    xt_t1 = A*xt_t;      %x_{t+1|t}
    Rxx_1 = A*Rxx*A' + Re;      %R^{xx}_{t+1|t}


    %Store the state vector
    Xsave(:,t) = xt_t;
end


diffInTotalRain = sum(y)- sum(sum(Xsave));

%%
plot(y,'g')
hold on
plot(sum(Xsave,1))
%%
% Rain_vector =Xsave(:).';
Rain_vector_Kassala = Xsave(:).';
figure,
plot(Rain_vector_Kassala)
hold on
plot(ElGeneina.rain);

%%
plotAcfPacfNormP(Rain_vector,nbroflags)
model_init = idpoly([1 0], [], []);
Rain_data = iddata(Rain_vector');
model_final_rain = pem(Rain_data,model_init);
res = resid(Rain_data,model_final_rain);
res = res.y(2:end);
plotAcfPacfNormP(res,nbroflags);
% save('myVariables.mat', 'Rain_vector');

%%
figure,
%plot(Rain_vector_Kassala-Rain_vector)
%%
save('myVariables2.mat','Rain_vector_Kassala')
