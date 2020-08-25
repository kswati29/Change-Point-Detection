import numpy as np
import pandas as pd
import statsmodels.api as sm

### Generate initial population
def initial_pop(n_chrom, size, p_zero, steps):
    """
    Description: generate initial population for GA
    
    Input:
        n_chrom: integer, number of chromsome 
        p_zero: [0,1], osomes
        size: integer, length of chromoprobability of a zero
        steps: minimum length of a segment
            
    Output:
        population: list of array, an array of 1s and 0s. No. of rows is n_chrom and number of columns is the number of observations
    (From Divya)
    """
    
    population = []
    for i in range(n_chrom):
            chromosome = np.ones(size) # Create an array of 1s
            mask = np.random.rand(size) < p_zero # Determine 0s locations
            chromosome[mask] = 0 # Replace 1s with 0s
            chromosome[0]=1 # Set first location as 1
            chromosome[-steps:]=0
            for val in range(len(chromosome)):
                if chromosome[val]==1:
                    chromosome[(val+1):(val+steps)]=0
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
        scores: array, the fitness score (MDL in this case)
    (From Swati)
    """
    
    scores = []
    for chromosome in population:
        n = len(chromosome) # n = length of chromosome

        pseudo_chromo = np.append(chromosome,[1]) # add 1 in the end to calculate segments
        pos = np.where(pseudo_chromo==1)[0] # list of change point positions
        
        y_seg_list = np.diff(pos) # list of length of segments of each chromosome
        k = len(pos)-2 # no. of change points excluding first and last position

        SSR_list = [] # store SSR of each segment in the chromosome
        for i in range(k+1): # exclude last CP
            # Extract the segment 
            x_seg= X.loc[pos[i]:pos[i+1]-1]
            y_seg = y.loc[pos[i]:pos[i+1]-1]
            #Model the data
            p_seg = sm.OLS(y_seg, sm.add_constant(x_seg)).fit()
            SSR = p_seg.ssr
            SSR_list.append(SSR)
        
        # Calculate MDL
        lastterm = (n/2) * np.log((1/n)*sum(SSR_list))
        #MDL = (np.log2(k+1)) + (k * np.log2(n)) + sum(np.log2(y_seg_list)) + lastterm
        MDL = (np.log2(k+1)) + sum(np.log2(y_seg_list)) + sum(np.log2(y_seg_list)) + lastterm
        scores.append(MDL)
        
    return(np.array(scores))
    


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
    
    ranking = pd.Series(scores).rank(method = 'first', ascending = False) # Rank from largest to smallest
    probabilities = ranking / ranking.sum() # higher score implies lower rank
    
    return probabilities


### Crossover
def crossover(population, probability, steps):
    """
    Description: Performs crossover operation
    
    Input:
        population: list of array, the current population
        probability: the sampling probabilities, in order of chromosomes in population
        steps: minimum length of a segment
        
    Output:
        child: array, the child chromosome
    (From Poonam)
    """
    
    # Choose parents
    parents = np.random.choice(range(len(population)), size = 2, replace = True, p = probability) # pick 2 chromosomes
    mom = population[parents[0]] # Extract mom chromosome
    dad = population[parents[1]] # Extract dad chromosome

    # Create new chromosome    
    child = np.zeros(len(mom))   # create child chromosome with all genes as zero initially
    i = 0
    while i < len(mom):
        if i == 0:
            child[i] = 1         # make first gene as 1
            i = i + steps      # skip ahead to end of segment
        else:
            child[i] = np.random.choice( [ mom[i], dad[i] ] ) # Randomly choose mom or dad gene in child chromosome (equal probability)
            if child[i] == 1:  # crossover gene is 1 
                i = i + steps   # jump ahead to t-steps if changepoint is found in crossover process
            else: i = i + 1    # Otherwise move to next gene
        
    # make last t_steps segment all zeroes
    child[(len(child) - steps) : ] = 0
    
    return np.array(child)


### Mutation
def mutation(population, probability, mut_prob, steps):
    """
    Description: Performs mutation operation
    
    Input: 
        population: list of array, the current population
        probability: the sampling probabilities, in order of chromosomes in population
        mut_prob: [0,1], probability of mutating a gene
        steps: minimum length of a segment
        
    Output:
        child: array, the child chromosome
    (From Poonam)
    """
    
    # Choose parent
    parent = np.random.choice(range(len(population)), size = 1, p = probability) # pick 1 chromosome
    parent = population[parent[0]]    
    
    # Randomly mutate gene
    child = np.zeros(len(parent))   # create child chromosome with all genes as zero initially
    
    i = 0
    while i < len(parent):
        if i == 0:
            child[i] = 1
            i = i + steps
        else:              
            child[i] = np.random.choice( [parent[i], np.abs(parent[i]-1)], p = [1-mut_prob, mut_prob]) # Randomly mutate gene
            if child[i] == 1: # current position is a change point
                i = i+ steps
            else: i = i + 1 # not change point, move on
                
    # make last t_steps segment all zeroes
    child[(len(child) - steps) : ] = 0     
    return np.array(child)


### Generate population
def generate(population, probability, cross_prob, mut_prob, steps):
    """
    Description: Creates the next generation
    
    Input:
        population: list of array, the current population
        probability: the sampling probabilities, in order of chromosomes in population
        cross_prob: [0,1], probabiliy of conducting crossover
        mut_prob: [0,1], probability of mutating a gene
        steps: minimum length of a segment
        
    Output:
        child: list of array, the child population
    
    """

    N = len(population)
    child = []
    
    for i in range(N):
        if np.random.rand() < cross_prob: # Conduct crossover
            child.append( crossover(population = population, probability = probability, steps = steps) )
        else:                             # Conduct mutation
            child.append( mutation(population = population, probability = probability, mut_prob = mut_prob, steps = steps) )

    return child


### Genetic Algorithm
def genetic_algorithm(X, y, N = 20, p_zero = 0.5, cross_prob = 0.9, mut_prob = 0.1, same_count = 5, max_iter = 50, steps = 5):
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
        steps: minimum length of a segment
        
    Output:
        best_model: chromosome, the model that gives the best fitness score
    """
    
    # Generates initial population
    current_population = initial_pop(n_chrom = N, size = X.shape[0], p_zero = p_zero, steps = steps)
    
    # Store best model
    scores = fitness_score(population = current_population, X = X, y = y)
    best_score = np.min(scores)
    best_model = current_population[np.argmin(scores)]
    
    itr = 1
    current_count = 1
    
    # Next generates and onward
    while itr <= max_iter and current_count <= same_count:
        
        # Generates next population
        probabilities = rank(scores = scores)
        child_population = generate(population = current_population, probability = probabilities, cross_prob = cross_prob, mut_prob = mut_prob, steps = steps)

        # Compare best chromosome
        scores = fitness_score(population = child_population, X = X, y = y)
        cur_best_score = np.min(scores)
        cur_best_model = child_population[np.argmin(scores)]
        print(cur_best_score)
        if cur_best_score < best_score: # Child is a better model, replace the overall best
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
    We will use two data sets: a simulated data set and Boston data set as an illustration
"""
### Simulated data set ###
np.random.seed(0)

# Generate x values
x = np.sort(np.random.rand(200))

# Generate y values
# We will use 4 pieces
# Piece 1: when 0 <= x < 0.25
# Pieces 2: when 0.25 <= x < 0.5
# Pieces 3: when 0.5 <= x < 0.75
# Pieces 4: when 0.75 <= x < 1

# Piece 1
x1 = x[x < 0.25]
y1 = 4 + 4*x1 + np.random.normal(size = len(x1))

# Piece 2
x2 = x[(x >= 0.25) & (x < 0.50)]
y2 = 4 - 4*x2 + np.random.normal(size = len(x2))

# Piece 3
x3 = x[(x >= 0.5) & (x < 0.75)]
y3 = -2.5 + 2.5*x3 + np.random.normal(size = len(x3))

# Piece 2
x4 = x[(x >= 0.75) & (x < 1)]
y4 = 2.5 - x4 + np.random.normal(size = len(x4))

# Combine to data frame
data = pd.DataFrame({'X': x, 'y': np.concatenate((y1, y2, y3, y4))})
# The true change point locations are at 50,94,158 (Python index)

### Set up Genetic Algorithm parameters ###
N = 100
p_zero = 0.9
cross_prob = 0.95
mut_prob = 0.05
same_count = 5
max_iter = 50
steps = 5

# Run the Genetic Algorithm
model = genetic_algorithm(data.X, data.y, N = N, p_zero = p_zero, cross_prob = cross_prob, 
                  mut_prob = mut_prob, same_count = same_count, max_iter = max_iter, steps = steps)


### Boston Data ###
# Load Data
BostonData = pd.read_csv("Boston.csv")
del BostonData['Unnamed: 0']

# Extract predictors and response
BostonData.sort_values('age', inplace = True)
BostonData.index = range(BostonData.shape[0])
y = BostonData.medv
X = BostonData.age

# Set up Genetic Algorithm parameters
N = 100
p_zero = 0.9
cross_prob = 0.95
mut_prob = 0.05
same_count = 5
max_iter = 50
steps = 5

# Run the Genetic Algorithm
model = genetic_algorithm(X, y, N = N, p_zero = p_zero, cross_prob = cross_prob, 
                  mut_prob = mut_prob, same_count = same_count, max_iter = max_iter, steps = steps)

