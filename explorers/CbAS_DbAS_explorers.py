from explorers.base_explorer import Base_explorer
import numpy as np

class DbAS_explorer(Base_explorer):

    def __init__(self, n_new_proposals, generator, Q = 0.9, n_convergence=10, backfill=True,
                 batch_size = 100, alphabet ="UCGA" , virtual_screen = 10,
                 mutation_rate = 0.2, path = "./simulations/", debug=False):
        #super(DbAS_explorer, self).__init__(batch_size, alphabet, virtual_screen, path, debug)  # for Python 2
        super().__init__(batch_size, alphabet, virtual_screen, path, debug)  # for Python 3
        self.explorer_type = 'DbAS'
        self.generator = generator
        self.Q = Q  # percentile used as the fitness threshold
        self.n_new_proposals = n_new_proposals
        self.backfill = backfill
        self.mutation_rate = mutation_rate
        self.all_proposals_ranked = []
        self.n_convergence = n_convergence  # assume convergence if max fitness doesn't change for n_convergence cycles


    def propose_samples(self):

        gamma = np.percentile(list(self.model.measured_sequences.values()), 100*self.Q)  # Qth percentile of current measured sequences
        initial_batch = [sequence for sequence in self.model.measured_sequences.keys() \
                         if self.model.measured_sequences[sequence] >= gamma]  # pick all measured sequences with fitness above the Qth percentile
        initial_weights = [1]*len(initial_batch)
        all_samples_and_weights = tuple((initial_batch, initial_weights))
        print('Starting a DbAS cycle...')
        print('Initial training set size: ', len(initial_batch))
        self.generator.get_model()
        self.generator.train_model(initial_batch, initial_weights)
        max_fitnesses = [np.max(self.model.measured_sequences.values())]
        count = 0  # total count of proposed sequences
        not_converged = True

        while (not_converged) and (count < self.batch_size * self.virtual_screen):

            # generate new samples using the generator (second argument is all existing measured seqs)
            proposals = self.generator.generate(self.batch_size, all_samples_and_weights[0])
            print(f'Proposed {len(proposals)} new samples')
            count += len(proposals)

            # calculate the scores of the new samples using the oracle
            scores = []
            for proposal in proposals:
                scores.append(self.model.get_fitness(proposal))
            print('Top score in proposed samples: ', np.max(scores))

            # set a new fitness threshold if the new percentile is higher than the current
            gamma_new = np.percentile(scores, self.Q*100)
            if gamma_new > gamma:
                gamma = gamma_new

            # calculate the weights for the proposed batch
            #weights = 1 - get_cdf(scores, gamma)  # TODO: implement get_cdf
            weights = [1 if score >= gamma else 0 for score in scores]

            # add proposed samples to the total sample pool
            all_samples = all_samples_and_weights[0] + proposals
            all_weights = all_samples_and_weights[1] + weights
            all_samples_and_weights = tuple((all_samples, all_weights))


            # update the generator with the new batch and its weights
            print('New training set size: ', len(all_samples_and_weights[0]))

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

        if self.backfill:  # TODO: maybe only sort by score, not by batch...?
            return self.all_proposals_ranked[:self.n_new_proposals]
        else:
            return [proposal for proposal in proposals if scores_dict[proposal] >= gamma]


def get_cdf():
    """
    returns the value of the cdf of an array x for each point of the array
    """
    num_bins = 20
    counts, bin_edges = np.histogram(data, bins=num_bins)
    cdf = np.cumsum(counts)
    pass


class CbAS_explorer(Base_explorer):

    def __init__(self, n_new_proposals=100, generator=None, Q = 0.9, n_convergence=10, backfill=True,
                 batch_size = 100, alphabet ="UCGA" , virtual_screen = 10,
                 mutation_rate = 0.2, path = "./simulations/", debug=False):
        #super(DbAS_explorer, self).__init__(batch_size, alphabet, virtual_screen, path, debug)  # for Python 2
        super().__init__(batch_size, alphabet, virtual_screen, path, debug)  # for Python 3
        self.explorer_type = 'DbAS'
        self.generator = generator
        self.Q = Q  # percentile used as the fitness threshold
        self.n_new_proposals = n_new_proposals
        self.backfill = backfill
        self.mutation_rate = mutation_rate
        self.all_proposals_ranked = []
        self.n_convergence = n_convergence  # assume convergence if max fitness doesn't change for n_convergence cycles


    def propose_samples(self):

        gamma = np.percentile(list(self.model.measured_sequences.values()), 100*self.Q)  # Qth percentile of current measured sequences
        initial_batch = [sequence for sequence in self.model.measured_sequences.keys() \
                         if self.model.measured_sequences[sequence] >= gamma]  # pick all measured sequences with fitness above the Qth percentile
        initial_weights = [1]*len(initial_batch)
        all_samples_and_weights = tuple((initial_batch, initial_weights))

        print('Starting a CbAS cycle...')
        print('Initial training set size: ', len(initial_batch))

        # store the initial model ar generator_0
        generator_0 = self.generator
        generator_0.get_model(seq_size=len(initial_batch[0]))
        generator_0.train_model(initial_batch, initial_weights)

        # this will be the current state of the generator
        self.generator.get_model(seq_size=len(initial_batch[0]))
        self.generator.train_model(initial_batch, initial_weights)

        max_fitnesses = [np.max(self.model.measured_sequences.values())]
        count = 0  # total count of proposed sequences
        not_converged = True

        while (not_converged) and (count < self.batch_size * self.virtual_screen):

            # generate new samples using the generator (second argument is all existing measured seqs)
            proposals = self.generator.generate(self.batch_size, all_samples_and_weights[0])
            print(f'Proposed {len(proposals)} new samples')
            count += len(proposals)

            # calculate the scores of the new samples using the oracle
            scores = []
            for proposal in proposals:
                scores.append(self.model.get_fitness(proposal))
            print('Top score in proposed samples: ', np.max(scores))

            # set a new fitness threshold if the new percentile is higher than the current
            gamma_new = np.percentile(scores, self.Q*100)
            if gamma_new > gamma:
                gamma = gamma_new

            # calculate the weights for the proposed batch
            log_probs_0 = generator_0.calculate_log_probability(proposals)
            log_probs_t = self.generator.calculate_log_probability(proposals)
            #print('log p 0', log_probs_0)
            #print('log p t', log_probs_t)
            #print('log0', log_probs_0)
            #print('logt', log_probs_t)
            #print('log0-logt', np.array(log_probs_0)-np.array(log_probs_t))
            weights_probs = [np.exp(logp0 - logpt) for (logp0, logpt) in list(zip(log_probs_0, log_probs_t))]
            weights_probs = np.nan_to_num(weights_probs)
            #print('weights_probs', weights_probs)
            weights_cdf = [1 if score >= gamma else 0 for score in scores]
            #print('weights_cdf', weights_cdf)
            weights = list(np.array(weights_cdf) * np.array(weights_probs))
            #print('weights', weights)


            # add proposed samples to the total sample pool
            all_samples = all_samples_and_weights[0] + proposals
            all_weights = all_samples_and_weights[1] + weights
            all_samples_and_weights = tuple((all_samples, all_weights))


            # update the generator with the new batch and its weights
            print('New training set size: ', len(all_samples_and_weights[0]))

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

        if self.backfill:  # TODO: maybe only sort by score, not by batch...?
            return self.all_proposals_ranked[:self.n_new_proposals]
        else:
            return [proposal for proposal in proposals if scores_dict[proposal] >= gamma]


# class CbAS_explorer(Base_explorer):
#
#     def __init__(self, generator, Q = 0.9, batch_size = 100, alphabet ="UCGA" , virtual_screen = 10,  path = "./simulations/" , debug=False):
#         super(DbAS_explorer, self).__init__(batch_size, alphabet, virtual_screen, path, debug)
#         self.explorer_type = 'CbAS'
#         self.generator = generator  # TODO: implement generator class
#         self.Q = Q  # percentile used as the fitness threshold

    # def propose_samples(self):
    #     generator_init = self.generator  # initial state of the generator
    #     model_init = self.model  # initial state of the oracle
    #     gamma = np.percentile()
    #     t = 0  # 'time'
    #     while gamma < desired_fitness:
    #
    #         # generate new samples using the generator
    #         new_batch_t = self.generator.generate_new_batch
    #
    #         # calculate the scores of the new samples using the oracle
    #         scores = self.model.get_fitness(new_batch_t)
    #
    #         # set a new fitness threshold if the new percentile is higher than the current
    #         gamma_new = np.percentile(scores, self.Q*100)
    #         if gamma_new > gamma:
    #             gamma = gamma_new
    #
    #         # calculate weights for the proposed batch
    #         log_pxt = np.sum(np.log(Xt_p) * Xt, axis=(1, 2))
    #         X0_p = generator_0.decoder_.predict(zt)
    #         log_px0 = np.sum(np.log(X0_p) * Xt, axis=(1, 2))
    #         w1 = np.exp(log_px0 - log_pxt)
    #         w2 = 1 - get_cdf(xt, gamma)
    #         weights = w1 * w2
    #
    #         t += 1
    #     return