from time import time
import pygad

# Define the available items
inventory = [
    {"item_name": "clock", "item_weight": 7, "item_value": 100},
    {"item_name": "painting-landscape", "item_weight": 7, "item_value": 300},
    {"item_name": "painting-portrait", "item_weight": 6, "item_value": 400},
    {"item_name": "radio", "item_weight": 2, "item_value": 40},
    {"item_name": "laptop", "item_weight": 5, "item_value": 500},
    {"item_name": "night-lamp", "item_weight": 6, "item_value": 70},
    {"item_name": "silver-cutlery", "item_weight": 1, "item_value": 100},
    {"item_name": "porcelain", "item_weight": 3, "item_value": 250},
    {"item_name": "bronze-figure", "item_weight": 10, "item_value": 300},
    {"item_name": "leather-handbag", "item_weight": 3, "item_value": 280},
    {"item_name": "vacuum-cleaner", "item_weight": 15, "item_value": 300},
]

# Define constants for the knapsack problem
MAX_WEIGHT = 25
DESIRED_VALUE = 1630


# Fitness function to evaluate solutions
def evaluate_fitness(ga_instance, solution, solution_idx):
    total_weight = sum(
        item["item_weight"] for item, included in zip(inventory, solution) if included
    )
    total_value = sum(
        item["item_value"] for item, included in zip(inventory, solution) if included
    )

    if total_weight > MAX_WEIGHT or total_value > DESIRED_VALUE:
        return 0
    return total_value


# Callback function for each generation
def generation_callback(ga_instance):
    best_solution = ga_instance.best_solution()
    best_solution_fitness = best_solution[1]
    if best_solution_fitness >= DESIRED_VALUE:
        ga_instance.keep_solving = False


# Variables to track performance
successful_solutions = 0
cumulative_success_time = 0

# Run the genetic algorithm multiple times
for run in range(10):
    start_time = time()

    ga_instance = pygad.GA(
        fitness_func=evaluate_fitness,
        gene_type=int,
        gene_space=[0, 1],
        num_generations=50,
        num_parents_mating=2,
        sol_per_pop=10,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        num_genes=len(inventory),
        on_generation=generation_callback,
    )

    ga_instance.run()

    best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()

    selected_items = [
        item["item_name"]
        for item, included in zip(inventory, best_solution)
        if included
    ]
    total_weight = sum(
        item["item_weight"]
        for item, included in zip(inventory, best_solution)
        if included
    )
    total_value = sum(
        item["item_value"]
        for item, included in zip(inventory, best_solution)
        if included
    )

    print(f"Final Solution: {selected_items}")
    print(f"Total Value: {total_value}")
    print(f"Total Weight: {total_weight}")
    elapsed_time = time() - start_time

    if best_fitness >= DESIRED_VALUE:
        successful_solutions += 1
        cumulative_success_time += elapsed_time

    ga_instance.plot_fitness().savefig(f"plot_{run + 1}.png")

print(f"Percentage of successful solutions: {successful_solutions / 10 * 100}%")
if successful_solutions > 0:
    print(
        f"Average time for successful solutions: {cumulative_success_time / successful_solutions} seconds"
    )
else:
    print("No successful solutions found.")
