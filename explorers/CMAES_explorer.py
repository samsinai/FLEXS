import numpy as np

from explorers.base_explorer import Base_explorer
from utils.sequence_utils import translate_one_hot_to_string

class CMAES_explorer(Base_explorer):
    # http://blog.otoro.net/2017/10/29/visual-evolution-strategies/ this is helpful
    
    def __init__(self,
                 batch_size=100,
                 alphabet="UCGA",
                 virtual_screen=10, 
                 path="./simulations/",
                 debug=False):
        super().__init__(batch_size,
                         alphabet,
                         virtual_screen,
                         path,
                         debug)
        self.explorer_type = f"CMAES"
        self.lam = 10
        
    def initialize_params(self):
        # to be called after set_model
        seq = list(self.model.measured_sequences.keys())[0]
        self.seq_len = len(seq)
        self.alphabet_len = len(self.alphabet)
        
        # we'll be working with one-hots
        N = self.seq_len * self.alphabet_len
        self.N = N
        self.mean = np.zeros(N)
        self.sigma = 1
        self.cov = np.identity(N)
        self.p_sigma = np.zeros(N)
        self.p_c = np.zeros(N)
        
        # approximation
        self.c_sigma = 3/N
        self.c_c = 4/N
        self.d_sigma = 1
        
        self.mu = self.lam//2
        self.mu_w = self.lam//4
        
    def convert_mvn_to_seq(self, mvn):
        # converts multivariate normal to one hot
        mvn = mvn.reshape((self.alphabet_len, self.seq_len))
        one_hot = np.zeros((self.alphabet_len, self.seq_len))
        amax = np.argmax(mvn, axis=0)
        
        for i in range(self.seq_len):
            one_hot[amax[i]][i] = 1
            
        return translate_one_hot_to_string(one_hot, self.alphabet)
        
    def _sample(self):
        samples = []
        for i in range(self.lam):
            x = np.random.multivariate_normal(self.mean, (self.sigma**2)*self.cov)
            seq = self.convert_mvn_to_seq(x)
            fitness = self.model.get_fitness(seq)
            
            samples.append((x, fitness))
        return samples
    
    def compute_new_mean(self, samples):
        s = np.zeros(self.mean.shape)
        
        # this is actually pretty bad, there's some recombination stuff going on here that I haven't done
        for i in range(self.mu):
            s += samples[i][0]
            
        s /= self.mu
        self.mean = s
        
    def expectation(self):
        return np.sqrt(self.N)*(1-1/(4*self.N)+1/(21*self.N**2))
        
    def update_isotropic_evolution_path(self, old_mean, new_mean):
        self.p_sigma = (1-self.c_sigma)*self.p_sigma + np.sqrt(1-(1-self.c_sigma)**2)*np.sqrt(self.mu_w)*np.linalg.inv(np.sqrt(self.cov))*(new_mean-old_mean)/self.sigma
        
    def update_step_size(self):
        s = self.c_sigma/self.d_sigma
        e = np.linalg.norm(self.p_sigma)/self.expectation()
        self.sigma = self.sigma*np.exp(s*(e-1))

    def p_sigma_indicator(self, squared=False):
        alpha = 1.5
        if squared:
            return int(np.linalg.norm(self.p_sigma**2) <= alpha*np.sqrt(self.N))
        return int(np.linalg.norm(self.p_sigma) <= alpha*np.sqrt(self.N))

    def update_anisotropic_evolution_path(self, old_mean, new_mean):
        self.p_c = (1-self.c_c)*self.p_c + self.p_sigma_indicator()*np.sqrt(1-(1-self.c_c)**2)*np.sqrt(self.mu_w)*(new_mean-old_mean)/self.sigma

    def update_covariance_matrix(self, samples, old_mean):
        c_1 = 2/(self.N**2)
        c_mu = self.mu_w/(self.N**2)
        c_s = (1 - self.p_sigma_indicator(squared=True))*c_1*self.c_c*(2-self.c_c)
        weighted_sum = sum([((samples[i][0]-old_mean)/self.sigma)*((samples[i][0]-old_mean)/self.sigma).T for i in range(len(samples))])/self.mu
        self.cov = (1-c_1-c_mu-c_s)*self.cov + c_1*(self.p_c*self.p_c.T) + c_mu

    def propose_samples(self):
        # in CMAES we _minimize_ an objective, so I'll conveniently reverse
        samples = sorted(self._sample(), key=lambda s: s[1], reverse=True)
            
        placeholder_mean = self.mean
        self.compute_new_mean(samples)
        self.update_isotropic_evolution_path(old_mean=placeholder_mean, new_mean=self.mean)
        self.update_anisotropic_evolution_path(old_mean=placeholder_mean, new_mean=self.mean)
        self.update_covariance_matrix(samples, old_mean=placeholder_mean)
        self.update_step_size()
        
        return [self.convert_mvn_to_seq(sample[0]) for sample in samples]
