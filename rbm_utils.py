from sklearn.neural_network import BernoulliRBM

class DeterministicRBM(BernoulliRBM):
    def transform(self, X):
        """Override to avoid Gibbs sampling and return hidden probabilities directly"""
        X = self._validate_data(X, accept_sparse='csr', reset=False)
        return self._mean_hiddens(X)
