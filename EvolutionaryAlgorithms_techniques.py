import numpy as np
import math
import Genetic_Main_Functions as Genetic
import matplotlib.pyplot as plt

### max_sigma should be 12, 2 and 160 for ackley, rastrigin and schwefel. this will make the boundries of x correct

def plot_result(itr, f1, f2 = None, f3 = None):
    X = np.array([i for i in range(1, itr+1)])
    Y1, Y2, Y3 = f1, f2, f3

    plt.plot(X, Y1, label='100 population')
    # plt.plot(X, Y2, label='200 population')
    # plt.plot(X, Y3, label='300 population')

    plt.xlabel('number of iterations')
    plt.ylabel('fitness on Ackley')
    plt.legend()
    plt.show()

    return


def self_adaptive_evolution_strategy(pop_size = 200, crossover = 'intermediate', step_type = 'uncorrelated-n-step', survival = 'truncated'):
    # setting parameters
    dim = 30
    population_size = pop_size
    itr = 1000
    fitnesses = []

    model = Genetic.EvolutionaryAlgorithm(dim=dim, algorithm='self-adaptive', population_size=population_size,crossover_type=crossover,
                                          mutation_type=step_type,survival_selection_method=survival, fitness_function='ackley')

    population, sigma = model.initial_population()
    iteration = itr
    while iteration > 0:
        a, b = model.find_best_answer(population)
        fitnesses.append(a)
        if iteration%100  == 0:
            print('iteration ', iteration, '. best fitness in population: ', np.round(a,4))
            # print(iteration)
        children, child_sig = model.crossover(population, sigma)
        children, child_sig = model.mutation(children, child_sig)
        population, sigma = model.generate_new_population(population, children, old_sig=sigma, new_sig=child_sig)
        iteration -= 1
    print(model.find_best_answer(population))

    return fitnesses



def self_adaptive_success_rule_evolution_strategy(pop_size, crossover = 'intermediate', survival = 'truncated'):
    # setting parameters
    dim = 30
    population_size = pop_size
    # depends on the function we are using and the range each function have
    itr = 1000
    fitnesses = []
    # it works better by intermediate crossover than the discrete
    model = Genetic.EvolutionaryAlgorithm(dim=dim, algorithm='self-adaptive-1/5-success-rule', population_size=population_size, crossover_type=crossover
                                          , mutation_type='uncorrelated-one-step', survival_selection_method=survival, fitness_function='ackley')
    population, sigma = model.initial_population()
    iteration = itr
    while iteration > 0:
        a, b = model.find_best_answer(population)
        fitnesses.append(a)
        if iteration % 100 == 0:
            print('iteration ', iteration, '. best fitness in population: ', np.round(a,4))
            # print(iteration)
        children, child_sig = model.crossover(population, sigma)
        children, child_sig = model.mutation(children, child_sig)
        population, sigma = model.generate_new_population(population, children, old_sig=sigma, new_sig=child_sig)
        iteration -= 1
    print(model.find_best_answer(population))
    return fitnesses


def differential_evolutionary(pop_size):
    # setting parameters
    dim = 30
    population_size = pop_size
    # depends on the function we are using and the range each function have
    itr = 1000
    fitnesses = []
    # it works better by intermediate crossover than the discrete
    model = Genetic.EvolutionaryAlgorithm(dim=dim, algorithm='differential-evolution',population_size=population_size, crossover_type='uniform'
                                          , mutation_type='differential',survival_selection_method='truncated', fitness_function='ackley')
    population = model.initial_population()
    # print(population.shape)
    iteration = itr
    while iteration > 0:
        a, b = model.find_best_answer(population)
        fitnesses.append(a)
        if iteration % 100 == 0:
            print('iteration ', iteration, '. best fitness in population: ', np.round(a, 4))
            # print(iteration)
        children = model.mutation(population)
        children = model.crossover(children)
        population = model.generate_new_population(population, children)
        iteration -= 1
    print(model.find_best_answer(population))
    return fitnesses


def pso_evolutionary_algorithm(pop_size):
    # setting parameters
    dim = 30
    population_size = pop_size
    # depends on the function we are using and the range each function have
    itr = 1000
    fitnesses = []
    # it works better by intermediate crossover than the discrete
    model = Genetic.EvolutionaryAlgorithm(dim=dim, algorithm='pso', population_size=population_size,crossover_type='uniform'
                                          , mutation_type='differential', survival_selection_method='truncated',
                                          fitness_function='ackley')
    population, velocity, bests = model.initial_population()
    iteration = itr
    while iteration > 0:
        a, b = model.find_best_answer(population)
        fitnesses.append(a)
        if iteration % 100 == 0:
            print('iteration ', iteration, '. best fitness in population: ', np.round(a, 4))
            # print(iteration)
        population, velocity, bests = model.mutation(population, velocity=velocity, bests=bests)
        iteration -= 1
    print(model.find_best_answer(population))
    return fitnesses


f1 = self_adaptive_evolution_strategy(100)
# f2 = self_adaptive_evolution_strategy(200)
# f3 = self_adaptive_evolution_strategy(300)

# f1 = self_adaptive_success_rule_evolution_strategy(200, crossover='discrete')
# f2 = self_adaptive_success_rule_evolution_strategy(200, crossover='discrete')
# f3 = self_adaptive_success_rule_evolution_strategy(300, crossover='discrete')

# f1 = differential_evolutionary(300)
# f2 = differential_evolutionary(200)
# f3 = differential_evolutionary(300)


# f1 = pso_evolutionary_algorithm(100)
# f2 = pso_evolutionary_algorithm(200)
# f3 = pso_evolutionary_algorithm(300)



plot_result(1000, f1)