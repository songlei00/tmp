import torch
import torch.nn as nn
from torch import Tensor
import gpytorch
from botorch import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal    
from botorch.acquisition import ExpectedImprovement


def identity_basis_fn(x):
    return x


class BayesianLinearModelBotorch(nn.Module):
    num_outputs = 1
    def __init__(self, dims, alpha, beta, train_X, train_Y, basis_fn):
        # likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # super(BayesianLinearModelBotorch, self).__init__(train_X, train_Y, likelihood)
        super(BayesianLinearModelBotorch, self).__init__()
        self.dims = dims
        self.alpha = alpha
        self.beta = beta
        self.train_X = train_X
        self.train_Y = train_Y
        self.basis_fn = basis_fn
        self._fit()

    def _fit(self):
        Phi = self.basis_fn(self.train_X)
        self.S_N_inv = self.alpha * torch.eye(Phi.shape[1]) + self.beta * Phi.T.mm(Phi)
        self.S_N = torch.linalg.inv(self.S_N_inv)
        self.m_N = self.beta * self.S_N.mm(Phi.T).mm(self.train_Y)
        print('m_N shape:', self.m_N.shape)
        print('S_N shape:', self.S_N.shape)

    def _posterior_predictive(self, x):
        Phi = self.basis_fn(x)
        y = (Phi @ self.m_N.float()).squeeze(-1)
        y_var = 1 / self.beta + torch.sum((Phi @ self.S_N.float()) * Phi, axis=-1)
        print('Phi shape:', Phi.shape)
        print('y shape:', y.shape)
        print('y var shape:', y_var.shape)
        return y, y_var
       
    def forward(self, x):
        y, y_var = self._posterior_predictive(x)
        if y_var.dim() == 2:
            covar = torch.cat([torch.diag(i).unsqueeze(0) for i in y_var], axis=0)
        else:
            covar = torch.diag(y_var)
        print('---')
        print('x shape', x.shape)
        print('y shape', y.shape)
        print('y var shape', y_var.shape)
        print('covar shape', covar.shape)
        print('---')
        return MultivariateNormal(y, covar)

    def posterior(self, x):
        mvn = self(x)
        posterior = GPyTorchPosterior(mvn=mvn)

        return posterior


class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_modulesss = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_modulesss(x)
        covar_x = self.covar_module(x)
        print('---')
        print('x shape:', x.shape)
        # print(torch.mean(x, axis=0))
        print('mean:', mean_x.shape)
        print(covar_x.dim())
        print(covar_x.shape)
        print('---')
        return MultivariateNormal(mean_x, covar_x)


def f(x):
    return torch.sin(x).sum() + torch.randn(1) * 0.1
train_X = torch.randn((10, 30)).double()
train_Y = []
for x in train_X:
    train_Y.append(f(x))
train_Y = torch.vstack(train_Y).double()
train_Y = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-6)

# model = SimpleCustomGP(train_X, train_Y)
model = BayesianLinearModelBotorch(30, 100, 100, train_X, train_Y, identity_basis_fn)
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# fit_gpytorch_model(mll)
print('after training')
EI = ExpectedImprovement(model, best_f=train_Y.max().item())

# test_x = torch.randn((10, 30))
# acqf_val = EI(test_x)
# print(acqf_val)
test_x = torch.randn((10, 1, 30))
acqf_val = EI(test_x)
print(acqf_val)