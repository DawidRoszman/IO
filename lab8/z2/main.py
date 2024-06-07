from math import cos, exp, sin
import numpy as np
import pygad


# Define the endurance function with meaningful parameter names
def compute_endurance(
    a: float, b: float, c: float, d: float, e: float, f: float
) -> float:
    return exp(-2 * (b - sin(a)) ** 2) + sin(c * d) + cos(e * f)


# Define the fitness function for the GA
def evaluate_fitness(instance, candidate, index):
    return compute_endurance(
        candidate[0],
        candidate[1],
        candidate[2],
        candidate[3],
        candidate[4],
        candidate[5],
    )


# Parameters for the genetic algorithm
pop_size = 50
num_genes = 6
num_generations = 50
mutation_rate = 0.15

# Initialize the population
initial_pop = np.random.rand(pop_size, num_genes)
initial_pop = np.clip(initial_pop, 0, 1)

# Configure the genetic algorithm
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=pop_size // 2,
    fitness_func=evaluate_fitness,
    sol_per_pop=pop_size,
    num_genes=num_genes,
    gene_type=np.float32,
    parent_selection_type="tournament",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=mutation_rate,
    mutation_by_replacement=True,
    random_mutation_min_val=0.0,
    random_mutation_max_val=1.0,
    initial_population=initial_pop,
)

# Run the genetic algorithm
ga_instance.run()

# Retrieve and display the best solution found
best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()

print(f"Parameters of the best solution: {best_solution}")
print(f"Fitness value of the best solution: {best_fitness}")

# Save the fitness plot
ga_instance.plot_fitness().savefig("fitness_plot.png")

# Example output:
# Parameters of the best solution: [0.13573422 0.13174489 0.99655145 0.99583983 0.08127417 0.03452893]
# Fitness value of the best solution: 2.8373140467769593
