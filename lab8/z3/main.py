from time import time
import matplotlib.pyplot as plt
import numpy as np
import pygad


# Define the maze
class Maze:
    def __init__(self):
        self.layout = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, "S", 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, "E", 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        self.start = (1, 1)
        self.end = (10, 10)

    def is_within_bounds(self, x, y):
        return 0 <= x < len(self.layout[0]) and 0 <= y < len(self.layout)

    def is_wall(self, x, y):
        return self.layout[y][x] == 1

    def is_end(self, x, y):
        return self.layout[y][x] == "E"

    def fitness(self, path):
        x, y = self.start
        fitness = 0

        for move in path:
            if move == 0:  # Move up
                y -= 1
            elif move == 1:  # Move down
                y += 1
            elif move == 2:  # Move left
                x -= 1
            elif move == 3:  # Move right
                x += 1

            if not self.is_within_bounds(x, y):
                fitness -= 100000
                break

            if self.is_wall(x, y):
                fitness -= 100000

            elif self.is_end(x, y):
                fitness += 10000
                break

            else:
                distance_to_exit = abs(x - self.end[0]) + abs(y - self.end[1])
                fitness += 10000 / (distance_to_exit + 1)

        return fitness


# Genetic Algorithm configuration class
class MazeSolverGA:
    def __init__(self, maze):
        self.maze = maze
        self.gene_space = [0, 1, 2, 3]
        self.num_genes = 30
        self.pop_size = 50
        self.num_parents_mating = 10
        self.num_generations = 1000
        self.parent_selection_type = "sss"
        self.keep_parents = -1
        self.crossover_type = "single_point"
        self.mutation_type = "random"
        self.mutation_percent_genes = 5

        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_function,
            sol_per_pop=self.pop_size,
            num_genes=self.num_genes,
            gene_space=self.gene_space,
            parent_selection_type=self.parent_selection_type,
            keep_parents=self.keep_parents,
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            mutation_percent_genes=self.mutation_percent_genes,
        )

    def fitness_function(self, model, solution, solution_idx):
        return self.maze.fitness(solution)

    def solve(self):
        iterations = 10
        total_time = 0

        for _ in range(iterations):
            start_time = time()
            self.ga_instance.run()
            end_time = time()
            total_time += end_time - start_time

        average_time = total_time / iterations
        print("Average time:", average_time, "seconds")

        start_time = time()
        self.ga_instance.run()
        end_time = time()
        execution_time = end_time - start_time

        best_solution, best_fitness, best_solution_idx = (
            self.ga_instance.best_solution()
        )

        print(f"Best solution parameters: {best_solution}")
        print(f"Best solution fitness: {best_fitness}")
        print("Execution time:", execution_time, "seconds")

        return best_solution

    def plot_solution(self, path):
        x, y = self.maze.start
        best_path = [(x, y)]
        for move in path:
            if move == 0:
                y -= 1
            elif move == 1:
                y += 1
            elif move == 2:
                x -= 1
            elif move == 3:
                x += 1
            best_path.append((x, y))

        maze_array = np.array(self.maze.layout)
        maze_array[maze_array == "S"] = 0
        maze_array[maze_array == "E"] = 0
        maze_array = maze_array.astype(int)

        plt.figure(figsize=(10, 8))
        plt.imshow(maze_array, cmap="binary", origin="upper")

        path_x, path_y = zip(*best_path)
        plt.plot(path_x, path_y, "r-", linewidth=2)

        plt.title("Maze Solution using Genetic Algorithm")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.xticks(np.arange(0, 12, 1))
        plt.yticks(np.arange(0, 12, 1))

        plt.savefig("plot.png")
        plt.show()


if __name__ == "__main__":
    maze = Maze()
    solver = MazeSolverGA(maze)
    best_solution = solver.solve()
    solver.plot_solution(best_solution)
