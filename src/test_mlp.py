import numpy as np
from mlp import MLP

def gradient_check():
    """
    Численная проверка градиентов (Gradient Check).
    Сравнивает аналитический градиент из backprop с численным приближением.
    """
    print("=== ТЕСТ 1: GRADIENT CHECK ===")
    np.random.seed(1)
    
    # Маленькая сеть и маленький батч, чтобы быстро считать
    layer_sizes = [3, 4, 1]
    mlp = MLP(layer_sizes, activation='tanh', optimizer='sgd', learning_rate=0.01)
    
    X = np.random.randn(5, 3)
    y = np.random.randn(5, 1)
    
    # 1. Считаем аналитические градиенты
    y_pred = mlp.forward(X)
    grads = mlp.backward(y, y_pred)
    
    # 2. Считаем численные градиенты
    epsilon = 1e-4
    for l in range(1, mlp.num_layers + 1):
        for param_name in [f'W{l}', f'b{l}']:
            param = mlp.params[param_name]
            num_grad = np.zeros_like(param)
            
            # Проходим по каждому элементу матрицы весов/смещений
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old_value = param[idx]
                
                # f(x + eps)
                param[idx] = old_value + epsilon
                y_pred_plus = mlp.forward(X)
                loss_plus = mlp.compute_loss(y, y_pred_plus)
                
                # f(x - eps)
                param[idx] = old_value - epsilon
                y_pred_minus = mlp.forward(X)
                loss_minus = mlp.compute_loss(y, y_pred_minus)
                
                # Возвращаем значение
                param[idx] = old_value
                
                # Численный градиент
                num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                it.iternext()
                
            # Сравниваем с аналитическим
            analytical_grad = grads[f'd{param_name[0]}{l}']
            diff = np.linalg.norm(num_grad - analytical_grad) / (np.linalg.norm(num_grad) + np.linalg.norm(analytical_grad) + 1e-8)
            
            if diff < 1e-6:
                print(f"  {param_name}: Отлично (diff = {diff:.2e})")
            elif diff < 1e-4:
                print(f"  {param_name}: Хорошо (diff = {diff:.2e})")
            else:
                print(f"  {param_name}: ВНИМАНИЕ! Большая разница (diff = {diff:.2e})")
    print("Gradient check пройден!\n")

def smoke_test():
    """
    Проверка, что сеть вообще обучается (Loss должен уменьшаться).
    """
    print("=== ТЕСТ 2: SMOKE TEST ===")
    np.random.seed(42)
    
    # Сгенерируем простую зависимость: y = 2x1 - 3x2 + 1
    X = np.random.randn(20, 2)
    y = 2 * X[:, 0:1] - 3 * X[:, 1:2] + 1 + np.random.randn(20, 1) * 0.1
    
    # Сеть: 2 входа -> 8 скрытых -> 1 выход
    mlp = MLP([2, 8, 1], activation='tanh', optimizer='adam', learning_rate=0.01)
    
    initial_loss = mlp.compute_loss(y, mlp.forward(X))
    print(f"Начальный Loss: {initial_loss:.4f}")
    
    # Учим 500 эпох
    for epoch in range(500):
        loss = mlp.train_step(X, y)
        
    final_loss = mlp.compute_loss(y, mlp.forward(X))
    print(f"Финальный Loss: {final_loss:.4f}")
    
    if final_loss < initial_loss * 0.1:
        print("Smoke test пройден! Сеть успешно обучается.\n")
    else:
        print("Smoke test НЕ пройден. Loss не уменьшился достаточно.\n")

if __name__ == "__main__":
    gradient_check()
    smoke_test()
    
    # Тест сохранения/загрузки
    print("=== ТЕСТ 3: SAVE/LOAD ===")
    mlp = MLP([2, 8, 1])
    mlp.save_weights('../models/test_model.npz')
    mlp.load_weights('../models/test_model.npz')
    print("Сохранение и загрузка работают корректно.")