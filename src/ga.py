import numpy as np

class GeneticAlgorithm:
    def __init__(self, 
                 pop_size=50, 
                 num_generations=100, 
                 crossover_prob=0.8, 
                 mutation_prob=0.1, 
                 mutation_sigma=0.1, 
                 elitism_count=2, 
                 tournament_size=3,
                 blx_alpha=0.5,
                 random_state=42):
        
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.blx_alpha = blx_alpha
        self.random_state = random_state

        self.population = None
        self.fitness_history = {'best': [], 'mean': []}
        self.best_solution = None
        self.best_fitness = -np.inf

    def _init_population(self, gene_length, bounds):
        """Инициализация популяции случайными значениями"""
        if bounds is None:
            # По умолчанию веса в диапазоне [-1, 1]
            self.population = np.random.uniform(-1, 1, (self.pop_size, gene_length))
        else:
            low, high = bounds
            self.population = np.random.uniform(low, high, (self.pop_size, gene_length))

    def _tournament_selection(self, fitness):
        """Турнирная селекция"""
        selected = []
        for _ in range(self.pop_size):
            # Выбираем случайных участников турнира
            indices = np.random.choice(self.pop_size, self.tournament_size, replace=False)
            # Победитель - тот, у кого больше fitness
            best_idx = indices[np.argmax(fitness[indices])]
            selected.append(self.population[best_idx].copy())
        return np.array(selected)

    def _blx_alpha_crossover(self, parent1, parent2, bounds=None):
        """Скрещивание BLX-alpha (Blend Crossover)"""
        if np.random.rand() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        # Вычисляем границы для смешивания
        c_min = np.minimum(parent1, parent2)
        c_max = np.maximum(parent1, parent2)
        diff = c_max - c_min
        
        # Расширяем границы на alpha
        lower_bound = c_min - self.blx_alpha * diff
        upper_bound = c_max + self.blx_alpha * diff
        
        # Генерируем потомков
        child1 = np.random.uniform(lower_bound, upper_bound)
        child2 = np.random.uniform(lower_bound, upper_bound)
        
        # Ограничиваем значения (если заданы)
        if bounds is not None:
            low, high = bounds
            child1 = np.clip(child1, low, high)
            child2 = np.clip(child2, low, high)
            
        return child1, child2

    def _gaussian_mutation(self, individual, bounds=None):
        """Гауссова мутация"""
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_prob:
                # Добавляем шум из нормального распределения
                noise = np.random.normal(0, self.mutation_sigma)
                individual[i] += noise
                
                # Ограничиваем значения
                if bounds is not None:
                    low, high = bounds
                    individual[i] = np.clip(individual[i], low, high)
        return individual

    def run(self, fitness_func, gene_length, bounds=None):
        """
        Основной цикл работы ГА.
        :param fitness_func: функция, принимающая 1D массив генов и возвращающая float (чем больше, тем лучше)
        :param gene_length: длина хромосомы (число весов)
        :param bounds: кортеж (min, max) для значений генов
        """
        np.random.seed(self.random_state)
        self._init_population(gene_length, bounds)
        
        for generation in range(self.num_generations):
            # 1. Оценка популяции
            fitness = np.array([fitness_func(ind) for ind in self.population])
            
            # 2. Сохраняем статистику
            best_idx = np.argmax(fitness)
            best_gen_fitness = fitness[best_idx]
            mean_gen_fitness = np.mean(fitness)
            
            self.fitness_history['best'].append(best_gen_fitness)
            self.fitness_history['mean'].append(mean_gen_fitness)
            
            if best_gen_fitness > self.best_fitness:
                self.best_fitness = best_gen_fitness
                self.best_solution = self.population[best_idx].copy()
                
            # 3. Элитизм (сохраняем лучших без изменений)
            sorted_indices = np.argsort(fitness)[::-1]
            new_population = [self.population[i].copy() for i in sorted_indices[:self.elitism_count]]
            
            # 4. Селекция
            parents = self._tournament_selection(fitness)
            
            # 5. Скрещивание и мутация
            while len(new_population) < self.pop_size:
                # Выбираем двух родителей случайным образом из отобранных
                p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
                
                child1, child2 = self._blx_alpha_crossover(p1, p2, bounds)
                
                new_population.append(self._gaussian_mutation(child1, bounds))
                if len(new_population) < self.pop_size:
                    new_population.append(self._gaussian_mutation(child2, bounds))
                    
            self.population = np.array(new_population)
            
        return self.best_solution, self.best_fitness