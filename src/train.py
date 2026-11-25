import os
import numpy as np
import time
import joblib
from mlp import MLP
from ga import GeneticAlgorithm
from ga_library import PyGADWrapper
from data import get_mini_batches

# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ГА
# ==========================================
def flatten_weights(mlp):
    """Превращает словарь весов MLP в одномерный массив (хромосому)"""
    flat_weights = []
    for l in range(1, mlp.num_layers + 1):
        flat_weights.append(mlp.params[f'W{l}'].flatten())
        flat_weights.append(mlp.params[f'b{l}'].flatten())
    return np.concatenate(flat_weights)

def unflatten_weights(flat_weights, mlp):
    """Восстанавливает словарь весов MLP из одномерного массива"""
    idx = 0
    for l in range(1, mlp.num_layers + 1):
        w_shape = mlp.params[f'W{l}'].shape
        b_shape = mlp.params[f'b{l}'].shape
        
        w_size = np.prod(w_shape)
        b_size = np.prod(b_shape)
        
        mlp.params[f'W{l}'] = flat_weights[idx : idx + w_size].reshape(w_shape)
        idx += w_size
        
        mlp.params[f'b{l}'] = flat_weights[idx : idx + b_size].reshape(b_shape)
        idx += b_size
    return mlp

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# ==========================================
# 1. ОБУЧЕНИЕ ГРАДИЕНТНЫМ МЕТОДОМ
# ==========================================
def train_mlp_gradient(X_train, y_train, X_val, y_val, 
                       layer_sizes=[8, 16, 8, 1], 
                       activation='relu', 
                       optimizer='adam', 
                       learning_rate=0.01, 
                       epochs=200, 
                       batch_size=32,
                       patience=20,
                       model_save_path='../models/gradient_model.npz'):
    
    print(f"\n--- Обучение градиентом (Архитектура: {layer_sizes}) ---")
    mlp = MLP(layer_sizes, activation=activation, optimizer=optimizer, learning_rate=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    start_time = time.time()
    for epoch in range(epochs):
        # Обучение по мини-батчам
        train_losses = []
        for X_batch, y_batch in get_mini_batches(X_train, y_train, batch_size, shuffle=True, random_state=epoch):
            loss = mlp.train_step(X_batch, y_batch)
            train_losses.append(loss)
            
        # Валидация
        y_val_pred = mlp.predict(X_val)
        val_loss = mlp.compute_loss(y_val, y_val_pred)
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            mlp.save_weights(model_save_path) # Сохраняем лучшую модель
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break
            
    train_time = time.time() - start_time
    
    # Оцениваем на валидации
    mlp.load_weights(model_save_path)
    val_rmse = calculate_rmse(y_val, mlp.predict(X_val))
    print(f"Градиент завершен. Время: {train_time:.2f}с. Val RMSE: {val_rmse:.4f}")
    
    return mlp, history, train_time

# ==========================================
# 2. ОБУЧЕНИЕ ГЕНЕТИЧЕСКИМ АЛГОРИТМОМ
# ==========================================
def train_mlp_ga(X_train, y_train, X_val, y_val,
                 layer_sizes=[8, 16, 8, 1],
                 activation='relu',
                 use_library=False, # Если True - используем PyGAD, иначе наш ГА
                 pop_size=50,
                 generations=50,
                 model_save_path='../models/ga_model.npz'):
    
    method_name = "PyGAD" if use_library else "Custom GA"
    print(f"\n--- Обучение {method_name} (Архитектура: {layer_sizes}) ---")
    
    # Создаем "пустую" сеть, чтобы узнать размерность весов
    mlp = MLP(layer_sizes, activation=activation)
    gene_length = len(flatten_weights(mlp))
    
    # Функция приспособленности (Fitness)
    def fitness_function(flat_weights):
        unflatten_weights(flat_weights, mlp)
        # Считаем ошибку только на валидационной выборке (чтобы избежать переобучения)
        # Нейросеть пытается МИНИМИЗИРОВАТЬ RMSE, а ГА МАКСИМИЗИРУЕТ fitness.
        # Поэтому возвращаем отрицательный RMSE.
        y_val_pred = mlp.predict(X_val)
        rmse = calculate_rmse(y_val, y_val_pred)
        return -rmse

    start_time = time.time()
    
    if use_library:
        ga_engine = PyGADWrapper(pop_size=pop_size, num_generations=generations)
    else:
        ga_engine = GeneticAlgorithm(pop_size=pop_size, num_generations=generations)
        
    best_weights, best_fitness = ga_engine.run(fitness_function, gene_length, bounds=(-3.0, 3.0))
    
    train_time = time.time() - start_time
    
    # Загружаем лучшие веса в модель и сохраняем
    unflatten_weights(best_weights, mlp)
    mlp.save_weights(model_save_path)
    
    val_rmse = -best_fitness
    print(f"{method_name} завершен. Время: {train_time:.2f}с. Val RMSE: {val_rmse:.4f}")
    
    return mlp, ga_engine.fitness_history, train_time

# ==========================================
# БЛОК ТЕСТИРОВАНИЯ (Запуск напрямую)
# ==========================================
if __name__ == "__main__":
    # Загружаем подготовленные данные из Этапа 2
    print("Загрузка данных...")
    data = np.load('../data/prepared_data.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # 1. Тестируем градиент (маленькое число эпох для проверки)
    train_mlp_gradient(
        X_train, y_train, X_val, y_val, 
        epochs=50, model_save_path='../models/test_grad.npz'
    )
    
    # 2. Тестируем наш ГА (маленькая популяция для проверки)
    train_mlp_ga(
        X_train, y_train, X_val, y_val, 
        use_library=False, pop_size=20, generations=20, 
        model_save_path='../models/test_ga_custom.npz'
    )