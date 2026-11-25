import numpy as np
import time
from ga import GeneticAlgorithm
from ga_library import PyGADWrapper

def sphere_function(individual):
    """
    Тестовая функция: f(x) = sum(x^2).
    Минимум в точке (0, 0, ...).
    Мы возвращаем отрицательное значение, так как ГА максимизирует fitness.
    """
    return -np.sum(np.square(individual))

def run_tests():
    gene_length = 5
    bounds = (-5.0, 5.0) # Ищем в диапазоне от -5 до 5
    
    print("=== ТЕСТ СВОЕГО ГА (GeneticAlgorithm) ===")
    start_time = time.time()
    custom_ga = GeneticAlgorithm(pop_size=50, num_generations=100, random_state=42)
    best_sol_custom, best_fit_custom = custom_ga.run(
        fitness_func=sphere_function, 
        gene_length=gene_length, 
        bounds=bounds
    )
    custom_time = time.time() - start_time
    
    print(f"Лучший результат: {best_sol_custom}")
    print(f"Лучший fitness: {best_fit_custom:.6f} (ближе к 0 - лучше)")
    print(f"Время выполнения: {custom_time:.4f} сек.\n")

    print("=== ТЕСТ БИБЛИОТЕКИ PYGAD ===")
    start_time = time.time()
    pygad_ga = PyGADWrapper(pop_size=50, num_generations=100, random_state=42)
    best_sol_pygad, best_fit_pygad = pygad_ga.run(
        fitness_func=sphere_function, 
        gene_length=gene_length, 
        bounds=bounds
    )
    pygad_time = time.time() - start_time
    
    print(f"Лучший результат: {best_sol_pygad}")
    print(f"Лучший fitness: {best_fit_pygad:.6f}")
    print(f"Время выполнения: {pygad_time:.4f} сек.\n")

    # Сравнение
    print("=== СРАВНЕНИЕ ===")
    if best_fit_custom > best_fit_pygad:
        print("Наш ГА нашел решение лучше (или равное).")
    else:
        print("PyGAD нашел решение лучше.")
        
    if custom_time < pygad_time:
        print("Наш ГА работает быстрее.")
    else:
        print("PyGAD работает быстрее.")

if __name__ == "__main__":
    run_tests()