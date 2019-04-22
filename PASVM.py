import numpy as np
from numpy.linalg import norm as l2
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.preprocessing import KernelCenterer


class PASVM(object):

    def __init__(self, C=1, relaxation="classic", coef0=1, degree=2, gamma=1.5, kernel_type=None):
        """
        Passive Agressive Support Vector Machine for Online Learning.
        """
        self.C = C
        self.relaxation = relaxation

        self.coef0 = 1
        self.degree = degree
        self.gamma = gamma
        self.kernel_type = kernel_type
        self.centerer = KernelCenterer()

    def c_(self, X):
        """
        Center the gram matrix
        """
        return self.centerer.fit_transform(X)

    def fit(self, X, y):
        if not hasattr(self, "W"):
            self.W = self._init_weights(X)

        if self.W.shape[0] != X.shape[1]:
            raise ValueError("Expecter to get X with {} features, got {} instead".format(
                X.shape[1], self.W.shape[0]))

        for i in range(X.shape[0]):

            x = X[i, :].reshape(1, -1)
            if self.kernel_type is not None:
                x = self.apply_kernel(x)
            loss = self._get_loss(x, y[i])
            tau = self._get_update_rule()(X, loss)

            self.W = self.W + tau*y[i]*x.reshape(-1, 1)

    def predict(self, X):
        if not hasattr(self, "W"):
            self.W = self._init_weights(X)
        return np.sign(np.dot(X, self.W))

    def _get_loss(self, X, y):

        loss = max(0, 1 - y*(np.dot(X, self.W)))
        return loss

    def _get_update_rule(self):

        def classic(X, loss):
            tau = loss/l2(X)
            return tau

        def first_relaxation(X, loss):
            tau = min(self.C, loss/l2(X))
            return tau

        def second_relaxation(X, loss):
            tau = loss/(l2(X) + (1/(2*self.C)))
            return tau

        mapping = {
            'classic': classic,
            'first': first_relaxation,
            'second': second_relaxation
        }

        return mapping[self.relaxation]

    def _init_weights(self, X):

        return np.random.randn(X.shape[1], 1)

    def apply_kernel(self, X):
        kernel_handler = {"rbf": self._apply_rbf,
                          "linear": self._apply_linear,
                          "poly": self. _apply_poly}
        return self.c_(kernel_handler[self.kernel_type](X))

    def _apply_linear(self, X):
        return linear_kernel(X)

    def _apply_poly(self, X):
        return polynomial_kernel(X, degree=self.degree, coef0=self.coef0, gamma=self.gamma)

    def _apply_rbf(self, X):
        return rbf_kernel(X, gamma=self.gamma)
