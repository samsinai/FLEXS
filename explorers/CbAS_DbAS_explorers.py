from explorers.base_explorer import Base_explorer
from utils.model_architectures import VAE
from utils.exceptions import GenerateError
from utils.sequence_utils import generate_random_mutant
import random
import numpy as np


class CbAS_explorer(Base_explorer):

    def __init__(self, generator=None, Q = 0.9, n_convergence=10, backfill=True,
                 batch_size=100, alphabet="UCGA", virtual_screen=10,
                 avg_mutations_per_sequence=2, path="./simulations/", path_vae="", debug=False):
        super().__init__(batch_size, alphabet, virtual_screen, path, debug)  # for Python 3
        self.generator = generator
        self.Q = Q  # percentile used as the fitness threshold
        self.n_new_proposals = self.batch_size
        self.backfill = backfill
        self.avg_mutations_per_sequence = avg_mutations_per_sequence
        self.all_proposals_ranked = []
        self.n_convergence = n_convergence  # assume convergence if max fitness doesn't change for n_convergence cycles
        self.explorer_type = f'CbAS_Q{self.Q}_generator{self.generator.name}'
        self.path_vae = path_vae


    def propose_samples(self):

        gamma = np.percentile(list(self.model.measured_sequences.values()), 100*self.Q)  # Qth percentile of current measured sequences
        initial_batch = [sequence for sequence in self.model.measured_sequences.keys() \
                         if self.model.measured_sequences[sequence] >= gamma]  # pick all measured sequences with fitness above the Qth percentile
        initial_weights = [1]*len(initial_batch)
        all_samples_and_weights = tuple((initial_batch, initial_weights))

        # find sequence with highest score in measured sequences
        self.top_sequence = initial_batch[0]
        top_score = self.model.measured_sequences[self.top_sequence]
        for sequence in initial_batch:
            if self.model.measured_sequences[sequence] > top_score:
                self.top_sequence = sequence
                top_score = self.model.measured_sequences[sequence]

        #print('Starting a CbAS cycle...')
        #print('Initial training set size: ', len(initial_batch))

        # this will be the current state of the generator
        self.generator.get_model(seq_size=len(initial_batch[0]), alphabet=self.alphabet)
        self.generator.train_model(initial_batch, initial_weights)

        # save the weights of the initial vae and save it as vae_0:
        # there are issues with keras model saving and loading, so we have to recompile it
        self.generator.vae.save(self.path_vae + 'vae_initial_weights.h5')
        generator_0 = VAE(batch_size=self.generator.batch_size,
                          latent_dim=self.generator.latent_dim,
                          intermediate_dim=self.generator.intermediate_dim,
                          epochs=self.generator.epochs,
                          epsilon_std = self.generator.epsilon_std,
                          beta=self.generator.beta,
                          validation_split=self.generator.validation_split,
                          min_training_size=self.generator.min_training_size,
                          avg_mutations_per_sequence=self.generator.avg_mutations_per_sequence,
                          verbose=False)
        generator_0.get_model(seq_size=len(initial_batch[0]), alphabet=self.alphabet)
        vae_0 = generator_0.vae
        vae_0.load_weights(self.path_vae + 'vae_initial_weights.h5')

        max_fitnesses = []  # keep track of max proposed fitnesses to check for convergence
        not_converged = True
        count = 0  # total count of proposed sequences

        while (not_converged) and (count < self.batch_size * self.virtual_screen):

            # generate new samples using the generator (second argument is a list of all existing measured and proposed seqs)
            proposals = []
            while len(proposals) == 0:
                try:
                    proposals = self.generator.generate(self.batch_size, all_samples_and_weights[0], self.top_sequence)
                    # print(f'Proposed {len(proposals)} new samples')
                    count += len(proposals)
                #except GenerateError as e:
                except:
                    #print(e.message)
                    print('Ending the CbAS cycle, returning existing proposals...')
                    if len(self.all_proposals_ranked) >= self.n_new_proposals:
                        return self.all_proposals_ranked[-self.n_new_proposals:]
                    else:
                        #print('got here')
                        random_mutants = []
                        for sample in initial_batch:
                            random_mutants.extend(list(set([generate_random_mutant(sample,
                                                                                   self.avg_mutations_per_sequence/len(sample),
                                                                                   alphabet=self.alphabet)
                                                            for i in range(self.n_new_proposals * 10)])))
                        for sample in initial_batch:
                            if sample in random_mutants:
                                random_mutants.remove(sample)
                        new_samples = random.sample(random_mutants, (self.n_new_proposals - len(self.all_proposals_ranked)))
                        #self.all_proposals_ranked.extend(new_samples)
                        return self.all_proposals_ranked + new_samples



            # calculate the scores of the new samples using the oracle
            scores = []
            for proposal in proposals:
                scores.append(self.model.get_fitness(proposal))
            #print('Top score in proposed samples: ', np.max(scores))

            # set a new fitness threshold if the new percentile is higher than the current
            gamma_new = np.percentile(scores, self.Q*100)
            if gamma_new > gamma:
                gamma = gamma_new

            # calculate the weights for the proposed batch
            log_probs_0 = self.generator.calculate_log_probability(proposals, vae=vae_0)
            log_probs_t = self.generator.calculate_log_probability(proposals)
            weights_probs = [np.exp(logp0 - logpt) for (logp0, logpt) in list(zip(log_probs_0, log_probs_t))]
            #weights_probs = np.nan_to_num(weights_probs)
            weights_cdf = [1 if score >= gamma else 0 for score in scores]
            weights = list(np.array(weights_cdf) * np.array(weights_probs))


            # add proposed samples to the total sample pool
            all_samples = all_samples_and_weights[0] + proposals
            all_weights = all_samples_and_weights[1] + weights
            all_samples_and_weights = tuple((all_samples, all_weights))


            # update the generator
            #print('New training set size: ', len(all_samples_and_weights[0]))
            self.generator.train_model(all_samples_and_weights[0], all_samples_and_weights[1])

            scores_dict = dict(zip(proposals, scores))
            self.all_proposals_ranked.extend(
                [proposal for proposal, score in sorted(scores_dict.items(), key=lambda item: item[1])])
            # all_proposals_ranked are in an increasing order or fitness, starting with the first batch

            # check if converged
            max_fitnesses.append(np.max(scores))
            if len(max_fitnesses) >= self.n_convergence:
                if len(set(max_fitnesses[-self.n_convergence:])) == 1:
                    not_converged = False
                    print('CbAS converged')

        self.all_proposals_ranked.reverse()

        if self.backfill:
            return self.all_proposals_ranked[:self.n_new_proposals]
        else:
            return [proposal for proposal in proposals if scores_dict[proposal] >= gamma]



class DbAS_explorer(Base_explorer):

    def __init__(self, generator=None, Q = 0.9, n_convergence=10, backfill=True,
                 batch_size=100, alphabet="UCGA", virtual_screen=10,
                 avg_mutations_per_sequence=2, path="./simulations/", debug=False):
        super().__init__(batch_size, alphabet, virtual_screen, path, debug)  # for Python 3
        self.generator = generator
        self.Q = Q  # percentile used as the fitness threshold
        self.n_new_proposals = self.batch_size
        self.backfill = backfill
        self.avg_mutations_per_sequence = avg_mutations_per_sequence
        self.all_proposals_ranked = []
        self.n_convergence = n_convergence  # assume convergence if max fitness doesn't change for n_convergence cycles
        self.explorer_type = f'DbAS_Q{self.Q}_generator{self.generator.name}'


    def propose_samples(self):

        gamma = np.percentile(list(self.model.measured_sequences.values()), 100*self.Q)  # Qth percentile of current measured sequences
        initial_batch = [sequence for sequence in self.model.measured_sequences.keys() \
                         if self.model.measured_sequences[sequence] >= gamma]  # pick all measured sequences with fitness above the Qth percentile
        initial_weights = [1]*len(initial_batch)
        all_samples_and_weights = tuple((initial_batch, initial_weights))

        # find sequence with highest score in measured sequences
        self.top_sequence = initial_batch[0]
        top_score = self.model.measured_sequences[self.top_sequence]
        for sequence in initial_batch:
            if self.model.measured_sequences[sequence] > top_score:
                self.top_sequence = sequence
                top_score = self.model.measured_sequences[sequence]

        #print('Starting a DbAS cycle...')
        #print('Initial training set size: ', len(initial_batch))

        # this will be the current state of the generator
        self.generator.get_model(seq_size=len(initial_batch[0]), alphabet=self.alphabet)
        self.generator.train_model(initial_batch, initial_weights)

        max_fitnesses = []  # keep track of max proposed fitnesses to check for convergence
        not_converged = True
        count = 0  # total count of proposed sequences

        while (not_converged) and (count < self.batch_size * self.virtual_screen):

            # generate new samples using the generator (second argument is a list of all existing measured and proposed seqs)
            proposals = []
            while len(proposals) == 0:
                try:
                    proposals = self.generator.generate(self.batch_size, all_samples_and_weights[0], self.top_sequence)
                    # print(f'Proposed {len(proposals)} new samples')
                    count += len(proposals)
                #except GenerateError as e:
                except:
                    #print(e.message)
                    print('Ending the DbAS cycle, returning existing proposals...')
                    if len(self.all_proposals_ranked) >= self.n_new_proposals:
                        return self.all_proposals_ranked[-self.n_new_proposals:]
                    else:
                        #print('got here')
                        random_mutants = []
                        for sample in initial_batch:
                            random_mutants.extend(list(set([generate_random_mutant(sample,
                                                                                   self.avg_mutations_per_sequence/len(sample),
                                                                                   alphabet=self.alphabet)
                                                            for i in range(self.n_new_proposals * 10)])))
                        for sample in initial_batch:
                            if sample in random_mutants:
                                random_mutants.remove(sample)
                        new_samples = random.sample(random_mutants, (self.n_new_proposals - len(self.all_proposals_ranked)))
                        #self.all_proposals_ranked.extend(new_samples)
                        return self.all_proposals_ranked + new_samples



            # calculate the scores of the new samples using the oracle
            scores = []
            for proposal in proposals:
                scores.append(self.model.get_fitness(proposal))
            #print('Top score in proposed samples: ', np.max(scores))

            # set a new fitness threshold if the new percentile is higher than the current
            gamma_new = np.percentile(scores, self.Q*100)
            if gamma_new > gamma:
                gamma = gamma_new

            # calculate the weights for the proposed batch
            weights_cdf = [1 if score >= gamma else 0 for score in scores]
            weights = list(np.array(weights_cdf))


            # add proposed samples to the total sample pool
            all_samples = all_samples_and_weights[0] + proposals
            all_weights = all_samples_and_weights[1] + weights
            all_samples_and_weights = tuple((all_samples, all_weights))


            # update the generator
            #print('New training set size: ', len(all_samples_and_weights[0]))
            self.generator.train_model(all_samples_and_weights[0], all_samples_and_weights[1])

            scores_dict = dict(zip(proposals, scores))
            self.all_proposals_ranked.extend(
                [proposal for proposal, score in sorted(scores_dict.items(), key=lambda item: item[1])])
            # all_proposals_ranked are in an increasing order or fitness, starting with the first batch

            # check if converged
            max_fitnesses.append(np.max(scores))
            if len(max_fitnesses) >= self.n_convergence:
                if len(set(max_fitnesses[-self.n_convergence:])) == 1:
                    not_converged = False
                    print('DbAS converged')

        self.all_proposals_ranked.reverse()

        if self.backfill:
            return self.all_proposals_ranked[:self.n_new_proposals]
        else:
            return [proposal for proposal in proposals if scores_dict[proposal] >= gamma]