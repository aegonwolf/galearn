from galearn import settings
from sklearn.model_selection import cross_val_score
import numpy as np

# having this here allows some functions to be called directly outside of simulate
rng = settings.rng
fitness_function = settings.fitness_function
estimator = settings.estimator
gene_pool = settings.gene_pool
gnp_window = settings.gnp_window
restrict_gnp = settings.restrict_gnp
p_outlier = settings.p_outlier


class Individual:
    def __init__(self, genes, fitness):
        self._genes = genes
        self._fitness = fitness
        self._fp = 0  # fitness_proportion to be used when selection == fp

    def __eq__(self, other):
        return self.genes == other.genes

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        return f"Individual with genes: {self._genes} and fitness:{self._fitness}"

    @property
    def genes(self):
        return self._genes

    @property
    def fitness(self):
        return self._fitness

    # may add cv = cv/skf as an option
    def set_fitness(self, cv = 3):
        self._fitness = get_fitness(settings.estimator(**self.genes), settings.fitness_function, cv = cv)

    def set_gene(self, gene, value):
        self._genes[gene] = value

    '''this function gets a gene from within the 
    valid window of genes'''

    def get_gene_from_window(self, gene):
        min_c = settings.gene_pool[gene].min()
        max_c = settings.gene_pool[gene].max()
        dist_1 = self._genes[gene] - min_c
        dist_2 = max_c - self._genes[gene]
        dist = min(dist_1, dist_2) * settings.gnp_window
        lb = self._genes[gene] - dist
        ub = self._genes[gene] + dist
        new_gene, alternate = rng.choice(gene_pool[gene][(gene_pool[gene] >= lb) & (gene_pool[gene] <= ub)], 2)
        return new_gene, alternate

    def mutate(self):
        """change gene with probability p_mutate"""
        gene = rng.choice(list(self._genes))
        if restrict_gnp and isinstance(settings.gene_pool[gene], float):
            # give chance of diversity = 1-p_mutate until p_outlier % chance
            if rng.random() < p_outlier:
                print(f"got an outlier")
                new_gene, alternate = rng.choice(settings.gene_pool[gene], 2)
            else:
                new_gene, alternate = self.get_gene_from_window(gene)
        else:
            new_gene, alternate = rng.choice(settings.gene_pool[gene], 2)
        # help make sure the gene get's mutated
        self._genes[gene] = alternate if new_gene == self._genes[gene] else new_gene
        return


class Population:
    """ creates a population with parameters = genepool of size = size"""
    def __init__(self, genepool, size=10):
        self._population = create_population(genepool, size)
        self._size = size

    # note that if several individuals have == best fitness anyone of them is returned in the sorted list
    @property
    def best_individual(self):
        """the individual with the highest fitness of the current population"""
        return self._population[0]

    @property
    def best_fitness(self):
        """the highest fitness achieved of the current population"""
        return self._population[0].fitness

    @property
    def population(self):
        """the current population: a list of individuals (models)"""
        return self._population

    @property
    def size(self):
        """the number of individuals in the current population"""
        return self._size

    def replace_generation(self, new_gen):
        """replaces the current generation with a new population"""
        new_gen.sort(reverse=True)
        self._population = new_gen


def set_settings(p_mutate, train_set, train_labels, scorer, model, params, restrict_gene_pool, gene_pool_window, cv):
    """global settings to be used across functions"""
    settings.p_outlier = 1 - p_mutate
    settings.X_train, settings.y_train = train_set, train_labels
    settings.fitness_function = scorer
    settings.estimator = model
    settings.gene_pool = params
    settings.restrict_gnp = restrict_gene_pool
    settings.gnp_window = gene_pool_window
    settings.cv = cv
    return


def simulate(params,
             scorer,
             iterations,
             model,
             train_set,
             train_labels,
             cv = 3,
             selection='truncation',
             p_cross=1.0,
             p_mutate=1.0,
             sim_ann=True,
             restrict_gene_pool=True,  # narrow genes i.e. finetune
             gene_pool_window=1.0,  # initial size of window
             decay=None,
             pop_size = 10,
             elitism=2):
    """Simulates natural selection of models with genetic algorithms and

    Parameters:
        scorer: sklearn scorer function, greater is better than
        iterations: the number of iterations to run the
        model: an sklearn API compatible model
        cv: number of cross validation folds
        selection: selection algorithm to be used, see more -> galearn.selection
        p_cross: probability of crossover
        p_mutate: probability of mutation
        sim_ann: Simulated Annealing, decay p_mutate and p_cross with time by rate decay
        restrict_gene_pool: restrict gene_pool size by rate = decay
        decay: used as exponential decay for probabilities and gene_pool_window
        pop_size: size of population
        elitism: the fixed number of individuals to make it into each iteration

    """
    set_settings(p_mutate, train_set, train_labels, scorer, model, params, restrict_gene_pool, gene_pool_window, cv)
    population = Population(settings.gene_pool, pop_size)
    best_fitness = population.best_fitness
    if decay is None:
        decay = 1 / iterations
    print(f"best initial fitness: {population.best_fitness}")
    for i in range(iterations):
        frac = int(1 - ((population.size - elitism) / population.size))
        new_gen = []
        breeding = select_breeding(population, selection, frac)

        for elite in range(elitism):
            new_gen.append(population.population[elite])

        # elitism to be implemented here
        while len(new_gen) < population.size:  # let population size oscillate +1 -1?
            parent_1, parent_2 = rng.choice(breeding, 2)  # possibility of selecting the same individual
            child_1, child_2 = breed(parent_1, parent_2, p_cross, p_mutate)
            new_gen.append(child_1)
            new_gen.append(child_2)
        # replace the previous generation
        population.replace_generation(new_gen)
        # are you better than the last?
        if best_fitness < population.best_fitness:
            diff = population.best_fitness - best_fitness
            best_fitness = population.best_fitness
            print(
                f"child {population.best_individual} with fitness {population.best_fitness}, which is {diff} better "
                f"than before")
        if sim_ann:
            p_cross = p_cross - p_cross * decay
            p_mutate = p_mutate - p_mutate * decay
            if settings.p_outlier > 0.1:
                settings.p_outlier = 1 - p_mutate
        if i % 50 == 0:
            print(f"p_cross is {p_cross}")
    # note if several individuals have same fitness anyone of them is returned
    return population.best_individual


def get_fitness(individual, fitness_fn, cv=3):
    X_train, y_train = settings.X_train, settings.y_train
    score = cross_val_score(individual, X_train, y_train, cv=cv, scoring=fitness_fn)
    return score.mean()


# creates a population of size size with parameters from gene_pool
def create_population(genepool, size=10):
    population = []
    for i in range(size):
        population.append(generate_parent(genepool))

    population.sort(reverse=True)
    return population


# generates a dictionary from the pool of genes
def generate_parent(genepool):
    parent = dict()
    for gene in genepool.keys():
        parent[gene] = rng.choice(genepool[gene])
    fitness = get_fitness(settings.estimator(**parent), settings.fitness_function)
    return Individual(parent, fitness)


# crossover two parents to create two children
# should not be called by itself because it doesn't set fitness
def crossover(parent_1, parent_2, child_1, child_2):
    # children are copies of parents by default
    genes = list(child_1.genes)  # make global to make more efficient!
    # select crossover point that is not on the end of the string
    start = rng.choice(range(len(genes) - 1))
    # no crossover happening
    if start == len(genes) - 1:
        return [child_1, child_2]
    cut = rng.choice(range(start, len(genes)))
    # no crossover happening
    if cut == start:
        return [child_1, child_2]
    # perform crossover
    for gene in genes[start:cut]:
        if isinstance(settings.gene_pool[gene],
                      float):  # introduce more diversity by modified crossover for continous values
            # could also solve this with algebra, but I like using the predefined gene_pool
            lower = parent_1[gene]
            higher = parent_2[gene]
            if parent_1[gene] > parent_2[gene]:
                lower = parent_2[gene]
                higher = parent_1[gene]

            new_gene_1, new_gene_2, = rng.choice(
                settings.gene_pool[gene][(settings.gene_pool[gene] >= lower) & (settings.gene_pool[gene] <= higher)], 2)
            child_1.set_gene(gene, new_gene_1)
            child_2.set_gene(gene, new_gene_2)
        else:
            child_1.set_gene(gene, parent_2.genes[gene])
            child_2.set_gene(gene, parent_1.genes[gene])

    return child_1, child_2


def breed(parent_1, parent_2, p_cross, p_mutate, cv =3):
    # check for recombination
    # if crossover happens at probability p then not crossover would happen at probability 1-p
    # rand() will draw a number larger than p_cross 1-p times
    # and a number < p_cross p times
    # children are copies of parents by default
    child_1, child_2 = Individual(parent_1.genes, parent_1.fitness), Individual(parent_2.genes, parent_2.fitness)
    if np.random.rand() < p_cross:
        # genes = list(child_1.genes)  # make global to make more efficient!
        child_1, child_2 = crossover(parent_1, parent_2, child_1, child_2)
    # mutate if p
    if np.random.rand() < p_mutate:
        child_1.mutate()
    if np.random.rand() < p_mutate:
        child_2.mutate()

    child_1.set_fitness(cv)
    child_2.set_fitness(cv)
    return child_1, child_2


def select_breeding(population, selection='truncation', frac=0.5):
    """selects the individuals that are viable for breeding and passing on their genese"""
    size = int(population.size * frac)
    if selection == 'truncation':
        cut = int(len(population.population) * frac)
        breeding = population.population[:cut]
        return breeding
    elif selection == 'fitness_proportionate' or selection == 'fp':
        return fp_selection(population, size)
    elif selection == 'tournament':
        return tournament_selection(population, size)
    elif selection == 'sus':
        return sus_selection(population, size)


# also elitism is almost unnecessary if tournament, almost!
def tournament_selection(pop, size):
    participants = [ind for ind in pop.population]
    breeding = []
    # could implement different rounds here
    # but I think that's almost the same as calling tournament different times with smaller sizes
    for i in range(size):
        a, b = rng.choice(participants, 2)
        if a > b:
            breeding.append(a)
            participants.remove(a)
        else:
            breeding.append(b)
            participants.remove(b)
    return breeding


# reverse tournament, eliminates need for elitism
# could use with parallelism
def rev_tournament_selection(pop, size):
    participants = [ind for ind in pop.population]
    breeding = [ind for ind in pop.population]
    num_eliminated = len(breeding) - size
    for i in range(num_eliminated):
        a, b = rng.choice(participants, 2)
        if a > b:
            breeding.remove(b)
        else:
            breeding.remove(a)
    return breeding


def fp_selection(pop, size):
    p = np.array([ind.fitness for ind in pop.population])
    total_fitness = p.sum()
    p = p / total_fitness
    # p = np.cumsum(p) nice alternative solution
    return rng.choice(pop.population, size=size, p=p).tolist()


# stochastic universal sampling
def sus_selection(pop, size):
    p = np.array([ind.fitness for ind in pop.population]).cumsum()
    total_fitness = np.array([ind.fitness for ind in pop.population]).sum()
    step = total_fitness / size
    start = rng.uniform(0, step)
    steps = [(start + i * step) for i in range(size)]
    i = 0
    breeding = []
    for s in steps:
        while p[i] > s and i < size:
            i = i + 1
            breeding.append(i)
    return breeding
