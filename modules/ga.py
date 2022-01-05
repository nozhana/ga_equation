'''Genetic Algorithm implemented to solve the following equation
    for given x values
    y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6'''

import numpy as np

# Mutation index
# mutidx = 4

# Creating the initial population
def new_pop(low, high, pop_size):
    return np.random.uniform(low=-4.0, high=4.0, size=pop_size)

# Calculate the fitness for individual solutions in population
def pop_fitness(xs, pop):
    return np.sum(pop * xs, axis=1)

# Calculate the fitness for an individual solution
# def sol_fitness(xs, sol):
#     pass

# Select the mating pool -- a.k.a 'parents' to mate
def sel_parents(pop, fitness, nparents):
    return pop[fitness.argsort()][-nparents:]

# Generate offspring using crossover breeding
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1idx = k % parents.shape[0]
        parent2idx = (k + 1) % parents.shape[0]
        offspring[k, :crossover_point] = parents[parent1idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2idx, crossover_point:]

    return offspring

# Mutate crossover offsprings
def mutate(offspring, range):
    mutidx = np.random.randint(offspring.shape[1])
    offspring[:, mutidx] += np.random.uniform(-1.,1.,offspring.shape[0])
    offspring[:, mutidx] = np.clip(offspring[:, mutidx], range[0], range[1])
    return offspring
