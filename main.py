'''Main code to generate solution generations to the following
    equation using the ga module
    y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6'''

from modules import ga
import numpy as np

# Values of x1 through x6
xvalues = np.array([4, -2, 3.5, 5, -11, -4.7])

# Number of weights to be optimized
nweights = 6

# Number of solutions per population
nsols = 8

# Solution range
low, high = -4.0, 4.0

# Number of generations to run
ngens = 50

# Number of mating parents to extract
nparents = 4

# Number of offsprings to be generated
noffsprings = nsols - nparents

new_pop = ga.new_pop(low, high, (nsols, nweights))
print('Initial population\n')
print(new_pop)
print('\n\n')

for gen in range(1, ngens+1):
    print('Generation ' + str(gen) + '\n\n')
    fitness = ga.pop_fitness(xvalues, new_pop)
    print('Fitness values gen ' + str(gen) + '\n')
    print(fitness)
    print('\n\n')
    parents = ga.sel_parents(new_pop, fitness, nparents)
    print('Parents gen ' + str(gen) + '\n')
    print(parents)
    print('\n\n')
    crossed = ga.crossover(parents, (noffsprings, nweights))
    print('Crossovers gen ' + str(gen) + '\n')
    print(crossed)
    print('\n\n')
    mutated = ga.mutate(crossed, (low, high))
    print('Mutates gen ' + str(gen) + '\n')
    print(mutated)
    print('\n\n')
    new_pop[:nparents] = parents
    new_pop[nparents:] = mutated
    print('Next gen population (gen ' + str(gen+1) + ')\n')
    print(new_pop)
    print('\n\n')
    print('Best fitness after gen ' + str(gen) + '\n')
    print(np.max(ga.pop_fitness(xvalues, new_pop)))
    print('\n\n')

print('BEST SOLUTION AFTER {} GENS:\n'.format(ngens))
print(new_pop[ga.pop_fitness(xvalues, new_pop).argsort()[-1]])
print('\n\n')
