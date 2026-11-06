%% PINN for delay logistic growth:  x'(t) = r*x(t)*(1 - x(t-tau)/K)
% Requires: Deep Learning Toolbox. Tested design for MATLAB R2025a.
% History:  h(t) = x(0)*exp(t/tau)  used whenever t - tau < 0 (delayed term only).
clear; clc; close all; rng(0);

%% ---------------- User settings ----------------
csvFile        = 'blowfly.csv';  % one column of counts, or [time in days, count]
dtSample       = 2;       % days between rows if the CSV has only counts (Nicholson data is bi-daily)
tMaxUse        = 400;     % use only data with t <= tMaxUse days (book Fig 5.18 shows the first 400 days; Inf = all)
estimateAll    = true;    % true: learn r,K,tau | false: tau=9, r*tau=2.1, learn K only
nFreezeTau     = 6000;    % iterations with tau/r frozen before unfreezing (if estimateAll)
physOnTrainOnly= false;   % true: collocation points only in training window
trainFrac      = 0.75;
numEpochs      = 10000;
nColloc        = 600;     % collocation points per iteration
wData          = 1;
wPhys          = 0.5;    % tune this (try 0.01 - 10)
nWarmup        = 3000;    % physics weight ramps linearly 0 -> wPhys over these iterations
useBest        = false;   % true: restore best-validation snapshot (may be an early one)
clipVal        = 1;       % elementwise gradient clipping threshold (guards against blow-up)
lrNet          = 1e-3;
lrPhys         = 5e-4;
nFourier       = 40;      % Fourier frequencies k = 1..nFourier (period T/k days)
hiddenUnits    = 48;
numLayers      = 4;
tau0 = 9;  rTau0 = 2.1;   % initial guesses / fallback values
rTauMin = pi/2 + 0.05;    % enforce r*tau > pi/2 (sustained oscillations); small margin

%% ---------------- Data ----------------
D = readmatrix(csvFile);
if size(D,2) >= 2
    t = D(:,1); x = D(:,2);
else
    x = D(:,1); t = dtSample*(0:numel(x)-1)';
end
t = t(:); x = x(:);
keep = t <= tMaxUse; t = t(keep); x = x(keep);
n = numel(t);
nTrain = floor(trainFrac*n);
tTrain = t(1:nTrain);      xTrain = x(1:nTrain);
tVal   = t(nTrain+1:end);  xVal   = x(nTrain+1:end);

% Normalisation: s = t/T in [0,1], u = x/xs
T  = max(t);
xs = max(xTrain);
sTrain = dlarray((tTrain/T)');  uTrain = dlarray((xTrain/xs)');
sVal   = dlarray((tVal/T)');    uVal   = dlarray((xVal/xs)');
u0 = xTrain(1)/xs;   % x(0), normalised: used in history h(t) = x(0)*exp(t/tau)
sMaxColloc = 1;
if physOnTrainOnly, sMaxColloc = tTrain(end)/T; end

%% ---------------- Network ----------------
nIn = 1 + 2*nFourier;
layers = featureInputLayer(nIn, 'Name','in');
for i = 1:numLayers
    layers = [layers
        fullyConnectedLayer(hiddenUnits, 'Name',"fc"+i)
        tanhLayer('Name',"tanh"+i)]; %#ok<AGROW>
end
layers = [layers; fullyConnectedLayer(1,'Name','out')];
net = dlnetwork(layers);

% Learnable physical parameters (log-parametrised to stay positive)
phys.logR   = dlarray(log(rTau0/tau0));
phys.logK   = dlarray(log(0.5));      % K/xs: half of training peak
phys.logTau = dlarray(log(tau0));

%% ---------------- Training ----------------
avgG = []; avgSq = []; avgGp = []; avgSqp = [];
good = struct('net',net,'phys',phys,'avgG',[],'avgSq',[],'avgGp',[],'avgSqp',[]);  % rollback point
bestVal = inf; bestNet = net; bestPhys = phys;
hist = struct('it',[],'train',[],'val',[],'r',[],'K',[],'tau',[]);
figure('Name','Training'); tl = tiledlayout(1,2);

for it = 1:numEpochs
    sC = dlarray(sMaxColloc*rand(1,nColloc));
    [loss, gNet, gPhys, parts] = dlfeval(@modelLoss, net, phys, sTrain, uTrain, ...
        sC, T, xs, nFourier, wData, wPhys*min(1, it/nWarmup), u0);

    % NaN/Inf guard: roll back to last good state and halve learning rates
    gradsOK = all(cellfun(@(g) all(isfinite(extractdata(g)),'all'), gNet.Value)) && ...
              all(structfun(@(g) all(isfinite(extractdata(g)),'all'), gPhys));
    if ~isfinite(double(extractdata(loss))) || ~gradsOK
        net = good.net; phys = good.phys;
        avgG = good.avgG; avgSq = good.avgSq; avgGp = good.avgGp; avgSqp = good.avgSqp;
        lrNet = lrNet/2; lrPhys = lrPhys/2;
        fprintf('  finite? loss: %d | gNet: %d | grad logR/logK/logTau: %d/%d/%d\n', ...
            isfinite(double(extractdata(loss))), ...
            all(cellfun(@(g) all(isfinite(extractdata(g)),'all'), gNet.Value)), ...
            isfinite(double(extractdata(gPhys.logR))), isfinite(double(extractdata(gPhys.logK))), ...
            isfinite(double(extractdata(gPhys.logTau))));
        fprintf('it %d: non-finite loss/gradient, rolled back. lrNet=%.2e lrPhys=%.2e\n', it, lrNet, lrPhys);
        continue
    end
    % Elementwise gradient clipping
    gNet  = dlupdate(@(g) max(min(g, clipVal), -clipVal), gNet);
    gPhys = dlupdate(@(g) max(min(g, clipVal), -clipVal), gPhys);

    % Freeze logic
    if ~estimateAll || it <= nFreezeTau
        gPhys.logR   = 0*gPhys.logR;
        gPhys.logTau = 0*gPhys.logTau;
    end
    if ~estimateAll
        phys.logR   = dlarray(log(rTau0/tau0));
        phys.logTau = dlarray(log(tau0));
    end

    [net, avgG, avgSq]    = adamupdate(net, gNet, avgG, avgSq, it, lrNet);
    [phys, avgGp, avgSqp] = adamupdate(phys, gPhys, avgGp, avgSqp, it, lrPhys);

    % Constraint: r*tau >= rTauMin  <=>  log r >= log(rTauMin) - log tau
    if estimateAll
        phys.logTau = min(max(phys.logTau, log(3)), log(30));            % tau in [3, 30] days
        phys.logR   = max(phys.logR, log(rTauMin) - phys.logTau);        % r*tau >= rTauMin
        phys.logR   = min(phys.logR, log(1));                            % r <= 1 per day
    end

    % Rollback checkpoint (only saved while everything is finite)
    if mod(it,50) == 0
        good = struct('net',net,'phys',phys,'avgG',avgG,'avgSq',avgSq,'avgGp',avgGp,'avgSqp',avgSqp);
    end

    if mod(it,100) == 0 || it == 1
        uPredVal = forward(net, embed(sVal, nFourier));
        valMSE = double(extractdata(mean((stripdims(uPredVal) - uVal).^2,'all')));
        r = exp(extractdata(phys.logR)); K = exp(extractdata(phys.logK))*xs;
        tau = exp(extractdata(phys.logTau));
        hist.it(end+1)=it; hist.train(end+1)=double(extractdata(parts.data));
        hist.val(end+1)=valMSE; hist.r(end+1)=r; hist.K(end+1)=K; hist.tau(end+1)=tau;
        if valMSE < bestVal, bestVal = valMSE; bestNet = net; bestPhys = phys; end
        if mod(it,500) == 0 || it == 1
            fprintf('it %5d | data %.3e | phys %.3e | val %.3e | r=%.4f K=%.1f tau=%.2f (r*tau=%.2f)\n', ...
                it, parts.data, parts.phys, valMSE, r, K, tau, r*tau);
        end
    end
end
nexttile(tl); semilogy(hist.it, hist.train, hist.it, hist.val); legend('train data MSE','val MSE'); xlabel('iteration'); grid on
nexttile(tl); yyaxis left; plot(hist.it, hist.tau, hist.it, hist.r*10); yyaxis right; plot(hist.it, hist.K);
legend('\tau','10 r','K'); xlabel('iteration'); title('Parameter estimates'); grid on

%% ---------------- Results ----------------
if useBest, net = bestNet; phys = bestPhys; end   % default: use final iteration
rEst = exp(extractdata(phys.logR)); KEst = exp(extractdata(phys.logK))*xs;
tauEst = exp(extractdata(phys.logTau));
fprintf('\nEstimated: r = %.4f, K = %.2f, tau = %.3f, r*tau = %.3f\n', rEst, KEst, tauEst, rEst*tauEst);

% Continuous solution estimate x(t) at arbitrary times
predictX = @(tq) double(extractdata(stripdims(forward(net, embed(dlarray(tq(:)'/T), nFourier)))))' * xs;
tq = linspace(0, T, 2000)';
figure; plot(tq, predictX(tq), "k.", "MarkerSize",3); hold on
plot(tTrain, xTrain, 'b.'); plot(tVal, xVal, 'r.');
xline(tTrain(end),'--'); legend('PINN x(t)','train','validation'); xlabel('t (days)'); ylabel('population'); grid on

tq = linspace(0, 400, 2000)';
writematrix([tq predictX(tq)], 'pinn_curve_wphys050.csv');

%% ---------------- Local functions ----------------
function feat = embed(s, nF)
    % s: 1xN dlarray (normalised time) -> formatted (1+2nF)xN features
    k = (1:nF)';
    arg = 2*pi*k*s;
    feat = dlarray([s; sin(arg); cos(arg)], 'CB');
end

function [loss, gNet, gPhys, parts] = modelLoss(net, phys, sD, uD, sC, T, xs, nF, wData, wPhys, u0)
    % ---- data loss
    uPredD = stripdims(forward(net, embed(sD, nF)));
    lossData = mean((uPredD - uD).^2, 'all');

    % ---- physics loss (normalised equation: du/ds = T*r*u*(1 - u(s-tau/T)/Kn))
    r   = exp(phys.logR);  Kn = exp(phys.logK);  tau = exp(phys.logTau);
    u   = stripdims(forward(net, embed(sC, nF)));
    dudt = dlgradient(sum(u,'all'), sC, 'EnableHigherDerivatives', true);

    sd   = sC - tau/T;                         % delayed normalised time
    mask = double(extractdata(sd) >= 0);       % 1 where network history available
    udNet = stripdims(forward(net, embed(max(sd, 0), nF)));
    uHist = u0 .* exp(min(sd,0)*T ./ tau);     % h(t-tau) = x(0)*exp((t-tau)/tau); only used where sd<0
    ud   = mask.*udNet + (1-mask).*uHist;

    res = dudt - T*r.*u.*(1 - ud./Kn);
    lossPhys = mean(res.^2, 'all');

    loss = wData*lossData + wPhys*lossPhys;
    [gNet, gPhys] = dlgradient(loss, net.Learnables, phys);
    parts.data = lossData; parts.phys = lossPhys;
end
