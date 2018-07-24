import sys
import copy
import numpy as np
from scipy.stats import truncnorm
from sklearn.neighbors.kde import KernelDensity
from batman.space import Space
import openturns as ot

sys.setrecursionlimit(10000)


class KdeSampler:
    """Kernel Density Estimation based sampler."""

    def __init__(self, sample=None, dim=2, n_sample_bin=1000, bw=1):
        """Sampler creation.

        A large bin of sample is used to draw new sample from using KDE.

        :param array_like sample: Sample to start from,
          shape (n_samples, n_features).
        :param int dim: Dimension of the parameter space.
        :param int n_sample_bin: Number of sample of the bin.
        :param float bw: Bandwidth of the KDE.
        """
        self.dim = dim
        if sample is None:
            self.space = [np.random.random_sample(self.dim)]
        else:
            self.space = sample

        self.n_samples = len(self.space)
        self.bw = bw
        self.bounds = np.array([[0] * self.dim, [1] * self.dim])

        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bw,
                                 metric='pyfunc',
                                 rtol=1e-4,
                                 metric_params={'func': self.metric_func})
        self.kde.fit(self.space)
        self.kde_ = copy.deepcopy(self.kde)

        # dists = [ot.Uniform(0, 1) for _ in range(self.dim)]
        # dists = ot.ComposedDistribution(dists)
        # lhs = ot.LHSExperiment(dists, n_sample_bin, True, True)
        # self.space_bin = np.array(lhs.generate())
        # # self.space_bin = np.array(ot.LowDiscrepancySequence(ot.SobolSequence(self.dim)).generate(n_sample_bin))
        # self.idx = list(range(n_sample_bin))

    def metric_func(self, x, other):
        """Inverse of Minkowsky with p=0.5."""
        p = 0.5

        # Bounds exclusion
        mask = np.logical_and(x >= self.bounds[0],
                              x <= self.bounds[1])
        if not np.all(mask):
            return 0

        # Non-rectangular domain
        # if not 0.5 < np.sum(x) < 1:
        #     return 0

        # Minkowsky
        dist = np.sum(abs(x - other) ** p) ** (1. / p)

        # Euclidean
        # dist = np.linalg.norm(x - other)

        # background = np.linalg.norm(x[1] - 0.8)
        # dist = 0
        # dist *= 1 / (background ) * 0.05

        # LHS constrain
        # if np.linalg.norm(x - other, -np.inf) <= 0.03 / (self.n_samples + 1):
        #     return 0

        # LHS + Limit influence
        # if (np.linalg.norm(x - other, -np.inf) <= 0.03 / (self.n_samples + 1)) and \
        #     (np.linalg.norm(x - other) <= 0.5 / (self.n_samples + 1) ** (1 / len(self.bounds[0]))):
        #     return 0

        return dist

    def pdf(self, x, kernel='gaussian'):
        """Scale PDF between 0 and 1."""
        pdf_base = np.exp(self.kde.score_samples(x))
        sigma_fin = self.bw / self.n_samples ** (1 / self.dim)
        pdf = 1 - (2 * np.pi) ** (self.dim / 2) * sigma_fin ** self.dim * pdf_base * self.n_samples  # gaussian
        # pdf = 1 - np.pi * sigma_fin ** 2 * f * self.n_samples  # tophat
        pdf[np.where(pdf < 0)] = 0

        return pdf

    def sample_kde(self, n_samples=1):
        """Generate random samples from the model.

        :param int n_samples: Number of samples to generate.
        :param return: List of samples.
        :rtype: array_like, shape (n_samples, n_features)
        """
        # proba = np.exp(self.kde.score_samples(self.space_bin))
        # proba = self.pdf(self.space_bin)
        # proba /= np.sum(proba)
        # idx = np.random.choice(self.idx, size=n_samples, p=proba)
        # return np.atleast_2d(self.space_bin[idx])

        def metropolis_accept(old, new):
            return np.log(np.random.uniform()) < new - old

        def proposal(x):
            lower, upper = -0.1, 1.1
            sigma = 0.3

            return np.array([truncnorm.rvs((lower - xi) / sigma,
                                           (upper - xi) / sigma,
                                           loc=xi, scale=sigma)
                             for xi in x]).reshape(1, -1)

        def metropolis(logp, n_samples, init):
            old = proposal(init)
            samples = []
            while len(samples) < n_samples:
                new = proposal(old)
                logp_old = logp(old)
                logp_new = logp(new)
                if metropolis_accept(logp_old, logp_new):
                    old = new
                    logp_old = logp_new

                if np.exp(logp_old) > 0:
                    samples.append(old)

            samples = np.atleast_2d(samples)[:n_samples].reshape(n_samples, -1)
            return samples

        # Restart
        # samples = np.random.random(self.bounds.shape[1]).reshape(1, -1)
        # while len(samples) < n_samples:
        #     samples_ = metropolis(self.kde.score_samples, n_samples // 1,
        #                           np.random.random(self.bounds.shape[1]))
        #     samples = np.concatenate([samples, samples_])

        # samples = metropolis(self.kde.score_samples, n_samples,
        #                      np.random.random(self.bounds.shape[1]))
        with np.errstate(divide='ignore', invalid='ignore'):
            samples = metropolis(lambda x: np.log(self.pdf(x)), n_samples,
                                 np.random.random(self.bounds.shape[1]))

        return samples

    def generate(self, n_samples=2):
        """Generate samples.

        Using the KDE, generate new samples following the PDF.
        The sample giving the best improvement in terms of discrepancy is kept.

        Update the KDE after each sample is added to the sampling.

        :param int n_samples: Number of samples to generate.
        :return: Sample.
        :rtype: array_like, shape (n_samples, n_features)
        """
        self.kde = copy.deepcopy(self.kde_)
        sample = list(copy.deepcopy(self.space))
        self.n_samples = len(sample)

        for _ in range(n_samples - 1):
            sample_ = self.sample_kde(500)

            self.sample_ = sample_
            self.kde_prev = copy.deepcopy(self.kde)

            # Normal strategy
            # disc = [ot.SpaceFillingPhiP(1000).evaluate(np.vstack([sample, s]))
            #         for s in sample_]

            # disc = [Space.discrepancy(np.vstack([sample, s]), method='WD')
            #         for s in sample_]

            disc = [ot.SpaceFillingC2().evaluate(np.vstack([sample, s]))
                    for s in sample_]

            # Subprojections
            # disc = [discrepancy_2D(np.vstack([sample, s]))
            #         for s in sample_]

            # Sobol consideration
            # disc = [ot.SpaceFillingC2().evaluate(np.concatenate([np.array(sample)[:, 0].reshape(-1, 1), np.array(s)[0].reshape(1, 1)]))
            #         for s in sample_]

            sample.append(sample_[np.argmin(disc)])

            # For constrain
            # disc = [ot.SpaceFillingMinDist().evaluate(np.vstack([sample, s]))
            #         for s in sample_]

            # Max probability point
            # disc = self.kde_.score_samples(sample_)

            # sample.append(sample_[np.argmax(disc)])

            self.n_samples = len(sample)
            self.kde.set_params(bandwidth=self.bw / self.n_samples ** (1 / 2),
                                metric_params={'func': self.metric_func})
            self.kde.fit(sample)

        return np.array(sample)
