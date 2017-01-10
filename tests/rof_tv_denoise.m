%% ROF TV Denoising

%%
% Add the toolbox.

addpath('../');
addpath('../toolbox/');

%%
% Load image
n = 256;
y = load_image('lena',n*2);
y = rescale(crop(y,n));
y = y + randn(n)*.06;

%%
% Display image
clf;
imageplot(clamp(y));

%%
% Optimization
lambda = .2;  % Regularization Parameter
K = @(x)grad(x); % Vectorial gradient
KS = @(x)-div(x); % Vectorial l1 norm

% We can put this problem as the minimization of F(K*x) + G(x)
Amplitude = @(u)sqrt(sum(u.^2,3));
F = @(u)lambda*sum(sum(Amplitude(u)));
G = @(x)1/2*norm(y-x,'fro')^2;

% The proximity operator of F is the vectorial soft thresholding
%Normalize = @(u)u./repmat(max(Amplitude(u),1e-10), [1 1 2]);
%ProxF = @(u,tau)repmat( perform_soft_thresholding(Amplitude(u),lambda*tau),[1 1 2]).*Normalize(u);
ProxF = @(u,tau)perform_soft_thresholding(Amplitude(u),lambda*tau);
ProxFS = compute_dual_prox(ProxF);


% record the progession of the functional
options.report = @(x)G(x) + F(K(x));

GradG = @(u)u-y;

%%
% Run FB on the primal ROF Model
L = 8;
options.method = 'fb';
[xFB,EFB] = perform_fb(y,ProxF,GradG,L,options);

%%
% Run FISTA on the primal ROF Model
options.method = 'fista';
[xFISTA,EFISTA] = perform_fb(y,ProxF,GradG,L,options);

%%
% Run ADMM on the primal ROF Model
ProxG = @(x,tau)(x+tau*y)/(1+tau); % The proximity operator of G
options.niter = 300;
[xADMM,EADMM] = perform_admm(y,K,KS,ProxFS,ProxG,options);


%%
% Run the ADMM algorithm
% options.niter = 300;
% [xAdmm,EAdmm] = perform_admm(y,K,KS,ProxFS,ProxG,options);
% clf;
% imageplot(xAdmm);

% Since the problem is strictly convex, we can use a FB scheme on the dual problem
% GradGS = @(x)x+y;
% L = 8;
% options.method = 'fista';
% [xFista,EFista] = perform_fb_strongly(y,K,KS,GradGS,ProxFS,L,options);
%
% options.method = 'fb';
% [xFB,EFB] = perform_fb_strongly(y,K,KS,GradGS,ProxFS,L,options);
%
% options.method = 'nesterov';
% [xNesterov,ENesterov] = perform_fb_strongly(y,K,KS,GradGS,ProxFS,L,options);


clf;
imageplot([y xFB xFISTA]);

% Compare energy decays
figure(2)
plot([EFB(:) EFISTA(:)]);
axis tight;
legend('FB','FISTA');
%axis([1 length(EFB) ]);%EFB(end)*.9 2000]);
