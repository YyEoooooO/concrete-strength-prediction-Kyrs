import pygad
import numpy as np

class PyGADWrapper:
    def __init__(self, 
                 pop_size=50, 
                 num_generations=100, 
                 crossover_prob=0.8, 
                 mutation_prob=0.1, 
                 random_state=42):
        
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_state = random_state
        self.ga_instance = None
        self.fitness_history = {'best': [], 'mean': []}

    def run(self, fitness_func, gene_length, bounds=None):
        """
        Запуск PyGAD.
        PyGAD ожидает функцию fitness_func(ga_instance, solution, solution_idx)
        """
        
        # Адаптируем нашу функцию под сигнатуру PyGAD
        def pygad_fitness(ga_instance, solution, solution_idx):
            return fitness_func(solution)

        # Настройка параметров
        if bounds is not None:
            init_range_low, init_range_high = bounds
            gene_space = {'low': init_range_low, 'high': init_range_high}
        else:
            init_range_low, init_range_high = -1.0, 1.0
            gene_space = None

        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=int(self.pop_size / 2),
            fitness_func=pygad_fitness,
            sol_per_pop=self.pop_size,
            num_genes=gene_length,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            gene_space=gene_space,
            crossover_probability=self.crossover_prob,
            mutation_probability=self.mutation_prob,
            mutation_type="random",
            mutation_num_genes=1,
            random_seed=self.random_state,
            suppress_warnings=True
        )

        # Запуск
        self.ga_instance.run()

        # Извлекаем лучший результат
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        
        # Сохраняем историю для графиков (PyGAD хранит её в своих атрибутах)
        self.fitness_history['best'] = self.ga_instance.best_solutions_fitness
        self.fitness_history['mean'] = self.ga_instance.average_fitness_per_generation if hasattr(self.ga_instance, 'average_fitness_per_generation') else [0]*self.num_generations

        return solution, solution_fitness