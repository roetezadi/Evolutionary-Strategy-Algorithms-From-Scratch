import numpy as np
import math
import Benchmark_Functions as Functions

class EvolutionaryAlgorithm:
    # creating necessary parameters
    def __init__(self, dim, algorithm, population_size, crossover_type, mutation_type, survival_selection_method, fitness_function, F = 0.2, fi1 = 0.1, fi2 = 0.3, w = 0.6):
        self.dim = dim
        self.algorithm = algorithm
        self.population_size = population_size
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.survival_selection_method = survival_selection_method
        self.fitness_function = fitness_function

        if algorithm == 'pso':
            self.u1 = np.array([[np.random.uniform(0,1) for i in range(self.dim)]  for i in range(self.population_size)])
            self.u2 = np.array([[np.random.uniform(0,1) for i in range(self.dim)] for i in range(self.population_size)])
            self.fi1, self.fi2, self.w = fi1, fi2, w
        elif algorithm == 'differential-evolution':
            self.F = F

        if fitness_function == 'ackley':
            self.max_sigma = 11
        elif fitness_function == 'rastrigin':
            self.max_sigma = 2
        elif fitness_function == 'schwefel':
            self.max_sigma = 160
        else:
            self.max_sigma = 1

    def initial_population(self):
        population = []
        if self.algorithm == 'self-adaptive':
            if self.mutation_type == 'uncorrelated-n-step':
                sigmas = []
                # min_sigma = 0.0125
                # max_sigma = 11 ==> this is for Ackley function
                mu = 0
                # creating population that have n genome with dim dimension: <x1,...,xn, sigma_1,...,sigma_n>
                for i in range(self.population_size):
                    tmp_p = []
                    tmp_s = []
                    for j in range(self.dim):
                        sigma = np.round(np.random.uniform(0, self.max_sigma), 2)
                        x = np.random.normal(mu, sigma, 1)[0]
                        tmp_p.append(x)
                        tmp_s.append(sigma)
                    population.append(tmp_p)
                    sigmas.append(tmp_s)

                # creating n genomes that have dim dimension with their distribution sigma
                population = np.array(population)
                sigmas = np.array(sigmas)

                return population, sigmas
            elif self.mutation_type == 'uncorrelated-one-step':
                sigma = 1
                # min_sigma = 0.0125
                # max_sigma = 11 ==> this is for Ackley function
                mu = 0
                # creating population that have n genome with dim dimension: <x1,...,xn, sigma_1,...,sigma_n>
                for i in range(self.population_size):
                    sigma = np.round(np.random.uniform(0, 5), 2)
                    x = np.random.normal(mu, self.max_sigma, self.dim)
                    population.append(x)

                # creating n genomes that have dim dimension with their distribution sigma
                population = np.array(population)

                return population, sigma

        elif self.algorithm == 'self-adaptive-1/5-success-rule':
            if self.mutation_type == 'uncorrelated-one-step':
                sigma = np.random.uniform(0, 1)
                mu = 0
                for i in range(self.population_size):
                    x = np.random.normal(mu, self.max_sigma, self.dim)
                    population.append(x)

                population = np.array(population)
                return population, sigma

        elif self.algorithm == 'differential-evolution':
            sigma = self.max_sigma
            mu = 0
            for i in range(self.population_size):
                x = np.random.normal(mu, sigma, self.dim)
                population.append(x)

            population = np.array(population)
            return population

        elif self.algorithm == 'pso':
            bests = []
            sigma = self.max_sigma
            mu = 0
            for i in range(self.population_size):
                x = np.random.normal(mu, sigma, self.dim)
                bests.append(x)
                population.append(x)

            population = np.array(population)
            bests = np.array(bests)

            f, best_in_p = self.find_best_answer(population)
            velocity = np.array(self.fi1*np.multiply(self.u1, bests - population) + self.fi2*np.multiply(self.u2, best_in_p - population))
            bests = population

            return population, velocity, bests

        else:
            print('Not Valid!')
            return

#################################################################################
################################## CROSSOVER ####################################
#################################################################################

    def crossover(self, population, sigmas = None):
        children = []
        new_sigmas = []
        incriminator = 0

        if self.algorithm == 'self-adaptive':
            if self.crossover_type == 'discrete' and self.mutation_type == 'uncorrelated-n-step':
                while incriminator < self.population_size:
                    # choosing parents for cross over. each 2 genomes in population are picked
                    parent1 = population[incriminator]
                    sigma1 = sigmas[incriminator]
                    incriminator += 1
                    parent2 = population[incriminator]
                    sigma2 = sigmas[incriminator]
                    incriminator += 1
                    # for each dimension we randomly choose parent1 or parent2 to copy its value in child dim
                    tmp = []
                    tmp_s = []
                    # check that we do the crossover or not
                    r = np.random.uniform(0, 1)
                    if r > 0.5:
                        for i in range(self.dim):
                            r2 = np.random.uniform(0, 1)
                            # parent1 is selected
                            if r2 > 0.5:
                                tmp.append(parent1[i])
                                tmp_s.append(sigma1[i])
                            # parent2 is selected
                            else:
                                tmp.append(parent2[i])
                                tmp_s.append(sigma2[i])
                        # adding the created child to the children's list
                        children.append(tmp)
                        new_sigmas.append(tmp_s)
                    else:
                        children.append(parent1)
                        new_sigmas.append(sigma1)
                        children.append(parent2)
                        new_sigmas.append(sigma2)

                children = np.array(children)
                new_sigmas = np.array(new_sigmas)

                return children, new_sigmas


            elif self.crossover_type == 'discrete' and self.mutation_type == 'uncorrelated-one-step':
                while incriminator < self.population_size:
                    # choosing parents for cross over. each 2 genomes in population are picked
                    parent1 = population[incriminator]
                    incriminator += 1
                    parent2 = population[incriminator]
                    incriminator += 1
                    # for each dimension we randomly choose parent1 or parent2 to copy its value in child dim
                    tmp = []
                    tmp_s = []
                    # check that we do the crossover or not
                    r = np.random.uniform(0, 1)
                    if r > 0.5:
                        for i in range(self.dim):
                            r2 = np.random.uniform(0, 1)
                            # parent1 is selected
                            if r2 > 0.5:
                                tmp.append(parent1[i])
                            # parent2 is selected
                            else:
                                tmp.append(parent2[i])
                        # adding the created child to the children's list
                        children.append(tmp)
                    else:
                        children.append(parent1)
                        children.append(parent2)

                children = np.array(children)
                new_sigmas = sigmas

                return children, new_sigmas


            elif self.crossover_type == 'intermediate' and self.mutation_type == 'uncorrelated-n-step':
                alpha = 0.5
                while incriminator < self.population_size:
                    # choosing parents for cross over. each 2 genomes in population are picked
                    parent1 = population[incriminator]
                    sigma1 = sigmas[incriminator]
                    incriminator += 1
                    parent2 = population[incriminator]
                    sigma2 = sigmas[incriminator]
                    incriminator += 1
                    # for each dimension we create the new child by intermediate strategy
                    tmp = alpha*parent1 + (1 - alpha)*parent2
                    tp = alpha*sigma1 + (1 - alpha)*sigma2
                    # adding the created child to the children's list
                    children.append(tmp)
                    new_sigmas.append(tp)

                children = np.array(children)
                new_sigmas = np.array(new_sigmas)

                return children, new_sigmas

            elif self.crossover_type == 'intermediate' and self.mutation_type == 'uncorrelated-one-step':
                alpha = 0.5
                while incriminator < self.population_size:
                    # choosing parents for cross over. each 2 genomes in population are picked
                    parent1 = population[incriminator]
                    incriminator += 1
                    parent2 = population[incriminator]
                    incriminator += 1
                    # for each dimension we create the new child by intermediate strategy
                    tmp = alpha * parent1 + (1 - alpha) * parent2
                    # adding the created child to the children's list
                    children.append(tmp)

                children = np.array(children)
                new_sigmas = sigmas

                return children, new_sigmas


        elif self.algorithm == 'self-adaptive-1/5-success-rule':
            if self.crossover_type == 'discrete':
                while incriminator < self.population_size:
                    # choosing parents for cross over. each 2 genomes in population are picked
                    parent1 = population[incriminator]
                    incriminator += 1
                    parent2 = population[incriminator]
                    incriminator += 1
                    # for each dimension we randomly choose parent1 or parent2 to copy its value in child dim
                    tmp = []
                    # check that we do the crossover or not
                    r = np.random.uniform(0, 1)
                    if r > 0.5:
                        for i in range(self.dim):
                            r2 = np.random.uniform(0, 1)
                            # parent1 is selected
                            if r2 > 0.5:
                                tmp.append(parent1[i])
                            # parent2 is selected
                            else:
                                tmp.append(parent2[i])
                        # adding the created child to the children's list
                        children.append(tmp)
                    else:
                        children.append(parent1)
                        children.append(parent2)

                children = np.array(children)
                new_sigmas = sigmas

                return children, new_sigmas

            elif self.crossover_type == 'intermediate':
                alpha = 0.5
                while incriminator < self.population_size:
                    # choosing parents for cross over. each 2 genomes in population are picked
                    parent1 = population[incriminator]
                    incriminator += 1
                    parent2 = population[incriminator]
                    incriminator += 1
                    # for each dimension we create the new child by intermediate strategy
                    tmp = alpha * parent1 + (1 - alpha) * parent2
                    # adding the created child to the children's list
                    children.append(tmp)

                children = np.array(children)
                new_sigmas = sigmas

                return children, new_sigmas

        elif self.algorithm == 'differential-evolution':
            # the crossover is in the mutation step
            while incriminator < self.population_size:
                # choosing parents for cross over. each 2 genomes in population are picked
                parent1 = population[incriminator]
                incriminator += 1
                parent2 = population[incriminator]
                incriminator += 1
                r = np.random.randint(0, self.dim)
                # for each dimension we create the new child by intermediate strategy
                tmp = np.concatenate((parent1[0:r],parent2[r:self.dim]), axis=0)
                tp = np.concatenate((parent2[0:r],parent1[r:self.dim]), axis=0)
                # adding the created child to the children's list
                children.append(tmp)
                children.append(tp)

            children = np.array(children)

            return children

        else:
            print('Error: the algorithm or the crossover method you have choose does not exist!')
            return population

    #################################################################################
    ################################## MUTATION ####################################
    #################################################################################


    def mutation(self, children, sigmas = None, velocity = None, bests = None):
        new_children = []
        new_sigmas = []
        N, M = children.shape[0], children.shape[1]
        if self.algorithm == 'self-adaptive':
            if self.mutation_type == 'uncorrelated-n-step':
                tau = 1/np.sqrt(np.sqrt(2)*self.dim)
                _tau = 1/np.sqrt(self.dim*2)
                # for all the child we apply mutation on them
                for k in range(N):
                    t1 = np.random.normal(0, 1)
                    s, c = [], []
                    for i in range(M):
                        t2 = np.random.normal(0, 1)
                        new_sigma = sigmas[k][i]*np.exp(t1*_tau + t2*tau)
                        new_x = children[k][i] + new_sigma*t2
                        c.append(new_x)
                        s.append(new_sigma)

                    new_children.append(c)
                    new_sigmas.append(s)

                new_children = np.array(new_children)
                new_sigmas = np.array(new_sigmas)

                return new_children, new_sigmas

            elif self.mutation_type == 'uncorrelated-one-step':
                tau = 1/np.sqrt(np.sqrt(self.dim))
                t1 = np.random.normal(0, 1)
                new_sigma = sigmas*np.exp(tau*t1)
                # for all the child we apply mutation on them
                for k in range(N):
                    c = []
                    for i in range(M):
                        t2 = np.random.normal(0, 1)
                        new_x = children[k][i] + new_sigma*t2
                        c.append(new_x)

                    new_children.append(c)
                new_children = np.array(new_children)
                return new_children, new_sigma

            elif self.mutation_type == 'correlated':
                pass

        elif self.algorithm == 'self-adaptive-1/5-success-rule':
            if self.mutation_type == 'uncorrelated-one-step':
                cnt_success = 0
                cnt_all = 0
                for k in range(N):
                    r = np.random.uniform(0, 1)
                    if r > 0.5:
                        new_x = np.random.normal(0, sigmas, self.dim)
                    else:
                        new_x = children[k]

                    if self.fitness_function == 'ackley':
                        if Functions.Ackley(new_x) > Functions.Ackley(children[k]):
                            cnt_success += 1
                    elif self.fitness_function == 'rastrigin':
                        if Functions.Rastrigin(new_x) > Functions.Rastrigin(children[k]):
                            cnt_success += 1
                    else:
                        if Functions.Schwefel(new_x) > Functions.Schwefel(children[k]):
                            cnt_success += 1
                    cnt_all += 1
                    new_children.append(new_x)
                new_children = np.array(new_children)

                c = 0.90
                if cnt_success/cnt_all > 0.2:
                    new_sigma = sigmas/c
                elif cnt_success/cnt_all < 0.2:
                    new_sigma = sigmas*c
                else:
                    new_sigma = sigmas

                return new_children, new_sigma

        elif self.algorithm == 'differential-evolution':
            F = self.F # 0.2
            for i in range(len(children)):
                indexes = [np.random.randint(0, len(children)) for i in range(3)]
                new_x = children[indexes[0]] + F * (children[indexes[1]] - children[indexes[2]])
                new_children.append(new_x)
                new_children.append(children[i])

            new_children = np.array(new_children)
            return new_children

        elif self.algorithm == 'pso':
            # fi1, fi2, w = 0.2, 0.5, 0.6 #OR 0.1, 0.3, 0.6 OR #0.01, 0.1, 0.6
            fi1, fi2, w = self.fi1, self.fi2, self.w
            f, best_in_pop = self.find_best_answer(children)
            for i in range(self.population_size):
                velocity[i] = w*velocity[i] + fi1*np.multiply(self.u1[i], (bests[i] - children[i])) + fi2*np.multiply(self.u2[i], (best_in_pop - children[i]))
                mutate_c = children[i] + velocity[i]
                if self.fitness_function == 'ackley':
                    if Functions.Ackley(mutate_c) < Functions.Ackley(children[i]):
                        bests[i] = mutate_c
                elif self.fitness_function == 'rastrigin':
                    if Functions.Rastrigin(mutate_c) < Functions.Rastrigin(children[i]):
                        bests[i] = mutate_c
                else:
                    if Functions.Schwefel(mutate_c) < Functions.Schwefel(children[i]):
                        bests[i] = mutate_c
                new_children.append(mutate_c)

            new_children = np.array(new_children)

            return new_children, velocity, bests

        else:
            print('Error: the algorithm or the mutation method you have choose does not exist!')
            return children

    #################################################################################
    ################################## NEW POPULATION ###############################
    #################################################################################

    def generate_new_population(self, old_pop, new_pop, old_sig = None, new_sig = None):

        if self.mutation_type == 'uncorrelated-n-step':
            if self.survival_selection_method == 'truncated':
                all_population = np.concatenate((old_pop, new_pop), axis=0)
                all_sigma = np.concatenate((old_sig, new_sig), axis=0)
                N = all_population.shape[0]
                if self.fitness_function == 'ackley':
                    fitnesses = np.array([Functions.Ackley(x) for x in all_population])
                elif self.fitness_function == 'rastrigin':
                    fitnesses = np.array([Functions.Rastrigin(x) for x in all_population])
                else:
                    fitnesses = np.array([Functions.Schwefel(x) for x in all_population])
                # we want to sort the array so we need to concatenate the population and sigmas with their fitness
                one_array = np.concatenate((np.concatenate((all_population, all_sigma), axis=1), fitnesses[:, None]), axis=1)
                one_array = np.array(sorted(one_array, key=lambda x: x[2*self.dim]))
                selected = one_array[0:self.population_size]
                survived_population, survived_sigmas = selected[:, 0:self.dim], selected[:, self.dim:self.dim*2]

                return survived_population, survived_sigmas
            else:
                all_population = np.concatenate((new_pop, old_pop), axis=0)
                all_sigma = np.concatenate((new_sig, old_sig), axis=0)
                survived_population = all_population[0:self.population_size, :]
                survived_sigmas = all_sigma[0:self.population_size, :]
                return survived_population, survived_sigmas


        elif self.mutation_type == 'uncorrelated-one-step':
            if self.survival_selection_method == 'truncated':
                all_population = np.concatenate((old_pop, new_pop), axis=0)
                N = all_population.shape[0]
                if self.fitness_function == 'ackley':
                    fitnesses = np.array([Functions.Ackley(x) for x in all_population])
                elif self.fitness_function == 'rastrigin':
                    fitnesses = np.array([Functions.Rastrigin(x) for x in all_population])
                else:
                    fitnesses = np.array([Functions.Schwefel(x) for x in all_population])
                # we want to sort the array so we need to concatenate the population and sigmas with their fitness
                one_array = np.concatenate((all_population, fitnesses[:, None]), axis=1)
                one_array = np.array(sorted(one_array, key=lambda x: x[one_array.shape[1]-1]))
                selected = one_array[0:self.population_size]
                survived_population, survived_sigmas = selected[:, 0:self.dim], old_sig

                return survived_population, survived_sigmas
            else:
                return old_pop, old_sig

        elif self.algorithm == 'differential-evolution':
            # in this algorithm we should choose the best one from ui and xi
            increminator = 0
            survived_population = []
            while increminator < len(new_pop):
                genome1 = new_pop[increminator]
                increminator += 1
                genome2 = new_pop[increminator]
                increminator += 1
                if self.fitness_function == 'ackley':
                    if Functions.Ackley(genome1) < Functions.Ackley(genome2):
                        survived_population.append(genome1)
                    else:
                        survived_population.append(genome2)
                elif self.fitness_function == 'rastrigin':
                    if Functions.Rastrigin(genome1) < Functions.Rastrigin(genome2):
                        survived_population.append(genome1)
                    else:
                        survived_population.append(genome2)
                else:
                    if Functions.Schwefel(genome1) < Functions.Schwefel(genome2):
                        survived_population.append(genome1)
                    else:
                        survived_population.append(genome2)

            survived_population = np.array(survived_population)

            return survived_population

        else:
            print('Not Valid!')
            return old_pop, old_sig

    #################################################################################
    ################################## BEST IN POPULATION ###########################
    #################################################################################


    def find_best_answer(self, population):
        if self.fitness_function == 'ackley':
            fitnesses = np.array([Functions.Ackley(x) for x in population])
        elif self.fitness_function == 'rastrigin':
            fitnesses = np.array([Functions.Rastrigin(x) for x in population])
        else:
            fitnesses = np.array([Functions.Schwefel(x) for x in population])

        all = np.concatenate((population, fitnesses[:, None]), axis=1)
        all = np.array(sorted(all, key=lambda x: x[self.dim]))
        min_value = all[0,self.dim]
        genome = all[0,0:self.dim]
        return min_value, genome