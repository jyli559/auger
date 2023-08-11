
import numpy as np
import scipy.stats as stats


class MultiArmedBandit:
    """Define a simple implementation of Multi-Armed Bandit with a Beta distribution.

    Based on https://peterroelants.github.io/posts/multi-armed-bandit-implementation/
    """


    def __init__(self, n_bandits, reshape_factor=1, minimize=False, boltzmann=False):
        self.n_bandits      = n_bandits
        self.alpha_priors   = np.ones(n_bandits)
        self.beta_priors    = np.ones(n_bandits)
        self.reshape_factor = reshape_factor # Higher reshape factor favors exploitation
        self.minimize       = minimize
        self.boltzmann      = boltzmann


    def choose_bandit(self):

        if self.boltzmann:
            probs = self.get_empirical_probs()
            self.chosen_bandit_idx = np.random.choice(self.n_bandits, p=np.exp(probs) / np.sum(np.exp(probs)))

        else:

            # Define prior distribution for each bandit
            self.bandit_priors = [stats.beta(alpha, beta)
                                  for alpha, beta in zip(self.alpha_priors, self.beta_priors)]

            # Sample a probability theta for each bandit
            theta_samples = [d.rvs(1) for d in self.bandit_priors]


            # Choose a bandit
            if self.minimize:
                self.chosen_bandit_idx = np.argmin(theta_samples)
            else:
                self.chosen_bandit_idx = np.argmax(theta_samples)

        return self.chosen_bandit_idx


    def update_posterior(self, observation):
        """Update the posterior of the current chosen bandit based on the observation (0 or 1)
        """

        assert 0 <= observation <= 1

        # Update posterior

        self.alpha_priors[self.chosen_bandit_idx] += observation * self.reshape_factor
        self.beta_priors[self.chosen_bandit_idx] += (1 - observation) * self.reshape_factor


    def get_empirical_probs(self):
        return np.array([(a / (a + b - 1)) for a, b in zip(self.alpha_priors, self.beta_priors)])




