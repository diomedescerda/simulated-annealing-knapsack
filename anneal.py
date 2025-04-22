from operator import index
import random

from mpmath import limit

from models import KnapsackItem
import math
import random
import matplotlib.pyplot as plt


class SimAnneal(object):
    def __init__(self, items: list[KnapsackItem], max_capacity, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.items = items
        self.N = len(items)
        self.max_capacity = max_capacity
        self.T = 1e15 if T == -1 else T
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 2e4 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.nodes = [i for i in range(self.N)]

        self.best_solution = None
        self.best_iteration = 0
        self.best_fitness = -float("Inf")
        self.fitness_list = []

    def is_valid(self, solution):
        total_weight = 0
        for quantity, item in zip(solution, self.items):
            if quantity < 0 or quantity > item.max_quantity:
                return False
            total_weight += quantity * item.weight
            if total_weight > self.max_capacity:
                return False
        return True

    # def initial_solution(self):
    #     """Greedy solution using value-to-weight ratio"""
    #     sorted_items = sorted(self.items,
    #                           key=lambda x: x.value / x.weight, reverse=True)
    #
    #     solution = [0] * self.N
    #     remaining_capacity = self.max_capacity
    #
    #     for item in sorted_items:
    #         idx = self.items.index(item)
    #         quantity = item.max_quantity
    #         while quantity > 0 and item.weight <= remaining_capacity:
    #             solution[idx] += 1
    #             remaining_capacity -= item.weight
    #             quantity -= 1
    #
    #     fitness = sum(item.value * count for item, count in zip(self.items, solution))
    #     return solution, fitness


    def random_initial_solution(self):
        """Greedy solution using value-to-weight ratio"""

        solution = [0] * self.N
        remaining_capacity = self.max_capacity
        limit = sum(item.max_quantity for item in self.items)
        for _ in range(limit):
            idx = self.items.index(random.choice(self.items))
            item = self.items[idx]
            quantity = item.max_quantity
            if quantity > 0 and item.weight <= remaining_capacity:
                solution[idx] += 1
                remaining_capacity -= item.weight
                quantity -= 1



        fitness = sum(item.value * count for item, count in zip(self.items, solution))
        return solution, fitness

    def generate_candidate(self, current_solution):
        for _ in range(100):
            candidate = current_solution[:]
            num_changes = random.randint(1, min(3, self.N))
            for _ in range(num_changes):
                idx = random.randint(0, self.N - 1)
                mutation = random.choice([-1, 1])
                candidate[idx] += mutation
            if self.is_valid(candidate):
                return candidate
        return current_solution

    def fitness(self, solution):
        total_weight = sum(item.weight * count for item, count in zip(self.items, solution))
        total_value = sum(item.value * count for item, count in zip(self.items, solution))

        if total_weight > self.max_capacity:
            return -1e9
        return total_value

    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        if candidate_fitness > self.cur_fitness:
            return 1.0
        try:
            return math.exp((candidate_fitness - self.cur_fitness) / self.T)
        except OverflowError:
            return 0.0

    def accept(self, candidate):
        candidate_fitness = self.fitness(candidate)
        # print(f"-|Current : {self.cur_fitness} -|candidate: {candidate_fitness}")
        # Accept always if better
        if candidate_fitness > self.cur_fitness:
            self.cur_solution = candidate
            self.cur_fitness = candidate_fitness
        else:
            # Accept probabilistically if worse
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_solution = candidate
                self.cur_fitness = candidate_fitness

        # Track best solution ever seen
        if candidate_fitness > self.best_fitness:
            self.best_fitness = candidate_fitness
            self.best_solution = candidate
            self.best_iteration = self.iteration

    def anneal(self):
        # Initialize with the greedy solution.
        self.cur_solution, self.cur_fitness = self.random_initial_solution()
        self.best_fitness, self.best_solution = self.cur_fitness, self.cur_solution
        self.greedy_fitness = self.cur_fitness  # ← Store it right away
        # print(f"Greedy solution: {self.cur_fitness}") testing purpose

        print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = self.generate_candidate(self.cur_solution)
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1

            self.fitness_list.append(self.cur_fitness)

        print("Found at iteration: ", self.best_iteration)
        print("Best fitness obtained: ", self.best_fitness)
        # improvement = 100 * (self.best_fitness - self.greedy_fitness) / (self.greedy_fitness)
        # print(f"Improvement over greedy heuristic: {improvement: .2f}%")

    def batch_anneal(self, times=10):
        """
        Execute simulated annealing algorithm `times` times, with random initial solutions.
        """
        for i in range(1, times + 1):
            print(f"Iteration {i}/{times} -------------------------------")
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.random_initial_solution()
            self.anneal()

    def plot_learning(self):
        """
        Plot the fitness through iterations.
        """
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()
