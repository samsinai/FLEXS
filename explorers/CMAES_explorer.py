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
        self.lam = self.batch_size
        self.round = 0
        
    def initialize_params(self):
        # to be called after set_model
        seq = list(self.model.measured_sequences.keys())[0]
        self.seq_len = len(seq)
        self.alphabet_len = len(self.alphabet)
        
        # we'll be working with one-hots
        N = self.seq_len * self.alphabet_len
        self.N = N
        self.mu = self.lam//2
        self.weights = [np.log(self.mu+0.5)-np.log(i) for i in range(1, self.mu+1)]
        self.weights /= sum(self.weights)
        self.mueff = sum(self.weights)**2/sum([w**2 for w in self.weights])
        
        self.c1 = 2/((self.N+1.3)**2+self.mueff)
        self.cc = (4+self.mueff/N)/(N+4+2*self.mueff/N)
        self.cs = (self.mueff+2)/(N+self.mueff+5)
        self.cmu = min(1-self.c1, 2*(self.mueff-2+1/self.mueff)/((N+2)**2+self.mueff))
        self.damp = 1 + 2*max(0, np.sqrt((self.mueff-1)/(N+1))-1) + self.cs
        
        self.mean = np.zeros(N)
        self.sigma = 0.3
        self.cov = np.identity(N)
        self.ps = np.zeros(N)
        self.pc = np.zeros(N)
        
        self.chiN = np.sqrt(N)*(1-1/(4*N)+1/(21*N**2))
        
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
        
        for i in range(self.mu):
            s += self.weights[i-1]*samples[i][0]
            
        self.mean = s
        
    def expectation(self):
        return np.sqrt(self.N)*(1-1/(4*self.N)+1/(21*self.N**2))
        
    def update_isotropic_evolution_path(self):
        self.ps = (1-self.cs)*self.ps \
            + np.sqrt(self.cs*(2-self.cs)*self.mueff)*np.linalg.inv(np.sqrt(self.cov))*(self.mean-self.old_mean)/self.sigma
        
    def update_step_size(self):
        self.sigma = self.sigma*np.exp((self.cs/self.damp)*(np.linalg.norm(self.ps)/self.chiN - 1))

    def ps_indicator(self):
        return int(np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.round/self.lam))/self.chiN < 1.4 + 2/(self.N+1))

    def update_anisotropic_evolution_path(self):
        self.pc = (1-self.cc)*self.pc \
            + self.ps_indicator()*np.sqrt(self.cc*(2-self.cc)*self.mueff)*(self.mean-self.old_mean)/self.sigma

    def update_covariance_matrix(self, samples):
        weighted_sum = sum([self.weights[i-1]*((samples[i][0]-self.old_mean)/self.sigma)*((samples[i][0]-self.old_mean)/self.sigma).T for i in range(self.mu)])
        
        self.cov = (1-self.c1-self.cmu)*self.cov \
            + self.c1*(self.pc*self.pc.T + (1-self.ps_indicator())*self.cc*(2-self.cc)*self.cov) \
            + self.cmu*weighted_sum

    def propose_samples(self):
        # in CMAES we _minimize_ an objective, so I'll conveniently reverse
        samples = sorted(self._sample(), key=lambda s: s[1], reverse=True)
            
        self.old_mean = self.mean
        self.compute_new_mean(samples)
        self.update_isotropic_evolution_path()
        self.update_anisotropic_evolution_path()
        self.update_covariance_matrix(samples)
        self.update_step_size()
        
        self.round += 1
        
        return [self.convert_mvn_to_seq(sample[0]) for sample in samples]
