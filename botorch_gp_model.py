import numpy as np


def identity_basis_function(x):
    return x

class BayesianLinearModel:
    def __init__(self, dims, alpha, beta, train_X, train_Y, basis_fn):
        self.dims = dims
        self.alpha = alpha
        self.beta = beta
        self.train_X = train_X
        self.train_Y = train_Y
        self.basis_fn = basis_fn

    def fit(self):
        self.m_N, self.S_N, self.S_N_inv = self._posterior(self.train_X, self.train_Y)

    def predict(self, test_X):
        return self._posterior_predictive(test_X, self.m_N, self.S_N)

    def _posterior(self, train_X, train_Y):
        Phi = self.basis_fn(train_X)
        S_N_inv = self.alpha * np.eye(Phi.shape[1]) + self.beta * Phi.T.dot(Phi)
        S_N = np.linalg.inv(S_N_inv)
        m_N = self.beta * S_N.dot(Phi.T).dot(train_Y)
        return m_N, S_N, S_N_inv

    def _posterior_predictive(self, test_X, m_N, S_N):
        print('X shape', test_X.shape)
        Phi_test = self.basis_fn(test_X)
        y = Phi_test.dot(m_N)
        print('phi tes shape:', Phi_test.shape)
        print(1 / self.beta)
        print(Phi_test.dot(S_N).shape)
        y_var = 1 / self.beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1, keepdims=True)
        print('var shape', y_var.shape)
        return y, y_var


def f(x):
    return np.sin(x).sum() + np.random.randn(1) * 0.1

dims = 30
train_X = np.random.randn(10, dims)
train_Y = []
for x in train_X:
    train_Y.append(f(x))
train_Y = np.vstack(train_Y)
train_Y = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-6)

model = BayesianLinearModel(dims, 100, 100, train_X, train_Y, identity_basis_function)
model.fit()

test_X = np.random.randn(3, dims)
y, y_var = model.predict(test_X)
print(y.shape)
print(y_var.shape)
# fit_gpytorch_model(mll)
# EI = ExpectedImprovement(model, best_f=train_Y.max().item())

# test_x = np.random.randn((1, dims))
# acqf_val = EI(test_x)
# print(acqf_val)

