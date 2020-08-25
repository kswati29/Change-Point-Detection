import numpy as np
import pandas as pd
import statsmodels.api as sm

### Generate initial population
def initial_pop(n_chrom, size, p_zero):
    """
    Description: generate initial population for GA
    
    Input:
        n_chrom: integer, number of chromosomes
        size: integer, length of chromosome 
        p_zero: [0,1], probability of a zero
        
    Output:
        population: list of array, an array of 1s and 0s. No of rows is n_chrom and number of columns is the features in X
    (From Divya)
    """
    
    population = []
    for i in range(n_chrom):
            chromosome = np.ones(size) # Create an array of 1s
            mask = np.random.rand(size) < p_zero # Determine 0s locations
            chromosome[mask] = 0 # Replace 1s with 0s
            population.append(chromosome) # Append new chromosome to population
    return population

### Evaluation
def fitness_score(population, X, y):
    """
    Description: calculates the fitness score of models
    
    Input:
        population: list of array, the current population
        X: dataframe, the predictors
        y: array, the response
        
    Output:
        scores: array, the fitness score (r-squared in this case)
    (From Divya)
    """
    
    scores = []
    for chromosome in population:
        X_intercept = sm.add_constant(X[X.columns[np.where(chromosome)]]) # Extract wanted columns, add intercept
        regression = sm.OLS(y, X_intercept).fit() # Fit linear regression
        scores.append(regression.rsquared_adj) # Extract adjusted r2 and append
    return scores


### Rank Probability
def rank(scores):
    """
    Description: Rank the chromosomes and return the sampling probabilities
    
    Input:
        scores: array, the fitness scores of the population
        
    Output:
        probabilities: the sampling probabilities, in order of chromosomes in population
    (From Divya and Swati)
    """
    
    ranking = pd.Series(scores).rank(method = 'first', ascending = True) # Rank from smallest to largest
    probabilities = ranking / ranking.sum()
    
    return probabilities


### Crossover
def crossover(population, probability):
    """
    Description: Performs crossover operation
    
    Input:
        population: list of array, the current population
        probability: the sampling probabilities, in order of chromosomes in population
        
    Output:
        child: array, the child chromosome
    (From Swati)
    """
    
    child = []
    
    # Choose parents
    parents = np.random.choice(range(len(population)), size = 2, replace = True, p = probability) # pick 2 chromosomes
    mom = population[parents[0]] # Extract mom chromosome
    dad = population[parents[1]] # Extract dad chromosome

    # Randomly choose mom or dad gene for each gene in child chromosome (equal probability)
    for i in range(len(mom)):
        child.append( np.random.choice( [ mom[i], dad[i] ] ) ) # Randomly choose between mom or dad
    
    return np.array(child)


### Mutation
def mutation(population, probability, mut_prob):
    """
    Description: Performs mutation operation
    
    Input: 
        population: list of array, the current population
        probability: the sampling probabilities, in order of chromosomes in population
        mut_prob: [0,1], probability of mutating a gene
        
    Output:
        child: array, the child chromosome
    (From Swati)
    """
    
    child = []

    # Choose parent
    parent = np.random.choice(range(len(population)), size = 1, p = probability) # pick 1 chromosome
    parent = population[parent[0]]

    # Randomly mutate gene
    for i in range(len(parent)):
        child.append(np.random.choice( [parent[i], np.abs(parent[i]-1)], p = [1-mut_prob, mut_prob]))
    
    return np.array(child)


### Generate population
def generate(population, probability, cross_prob, mut_prob):
    """
    Description: Creates the next generation
    
    Input:
        population: list of array, the current population
        probability: the sampling probabilities, in order of chromosomes in population
        cross_prob: [0,1], probabiliy of conducting crossover
        mut_prob: [0,1], probability of mutating a gene
    
    Output:
        child: list of array, the child population
    
    """

    N = len(population)
    child = []
    
    for i in range(N):
        if np.random.rand() < cross_prob: # Conduct crossover
            child.append( crossover(population = population, probability = probability) )
        else:                             # Conduct mutation
            child.append( mutation(population = population, probability = probability, mut_prob = mut_prob) )

    return child


### Genetic Algorithm
def genetic_algorithm(X, y, N = 20, p_zero = 0.5, cross_prob = 0.9, mut_prob = 0.1, same_count = 5, max_iter = 50):
    """
    Description: The overall genetic algorithm
    
    Input:
        X: dataframe, set of predictors
        y: series, the response 
        n_chrom: integer, number of chromosomes
        p_zero: [0,1], probability of a zero in inital chromosome generation
        cross_prob: [0,1], probabiliy of conducting crossover
        mut_prob: [0,1], probability of mutating a gene
        same_count: integer, number of consecutive best model to compare
        max_iter: integer, maximum number of iterations to run
        
    Output:
        best_model: chromosome, the model that gives the best fitness score
    """
    
    # Generates initial population
    current_population = initial_pop(n_chrom = N, size = X.shape[1], p_zero = 0.5)
    
    # Store best model
    scores = fitness_score(population = current_population, X = X, y = y)
    best_score = np.max(scores)
    best_model = current_population[np.argmax(scores)]
    
    itr = 1
    current_count = 1
    
    # Next generates and onward
    while itr <= max_iter and current_count <= same_count:
        
        # Generates next population
        probabilities = rank(scores = scores)
        child_population = generate(population = current_population, probability = probabilities, cross_prob = cross_prob, mut_prob = mut_prob)

        # Compare best chromosome
        scores = fitness_score(population = child_population, X = X, y = y)
        cur_best_score = np.max(scores)
        cur_best_model = child_population[np.argmax(scores)]
        
        if cur_best_score > best_score: # Child is a better model, replace the overall best
            best_score = cur_best_score
            best_model = cur_best_model
            current_count = 1           # Reset current count
        else:                           # Child is not better
            current_count += 1

        
        # Update current population
        current_population = child_population
        itr += 1

    return best_model


################################################################################################
"""
Example Code: 
    We will use the Boston data set as an illustration.
"""
# Load data
BostonData = pd.read_csv("Boston.csv")
del BostonData['Unnamed: 0']

# Extract predictors and response
X = BostonData.drop('medv', axis = 1)
y = BostonData.medv

# Set up Genetic Algorithm parameters
N = 20
p_zero = 0.5
cross_prob = 0.9
mut_prob = 0.1
same_count = 5
max_iter = 50
 
# Run the Genetic Algorithm
model = genetic_algorithm(X, y, N = N, p_zero = p_zero, cross_prob = cross_prob, 
                  mut_prob = mut_prob, same_count = same_count, max_iter = max_iter)

# Check out model
X_intercept = sm.add_constant(X[X.columns[np.where(model)]]) 
sm.OLS(y, X_intercept).fit().rsquared_adj


# Adj R^2 on full model
X_full = sm.add_constant(X)
sm.OLS(y, X_full).fit().rsquared_adj
