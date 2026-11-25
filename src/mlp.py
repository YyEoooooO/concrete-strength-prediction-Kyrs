import numpy as np
import os

# ==========================================
# 1. ФУНКЦИИ АКТИВАЦИИ И ИХ ПРОИЗВОДНЫЕ
# ==========================================
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

ACTIVATIONS = {
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, linear_derivative)
}

# ==========================================
# 2. ОПТИМИЗАТОРЫ
# ==========================================
class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            grad_key = f'd{key}' # <-- ИСПРАВЛЕНИЕ: формируем ключ (например, 'dW1')
            params[key] -= self.lr * grads[grad_key]

class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = {}

    def update(self, params, grads):
        if not self.v:
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key])
                
        for key in params.keys():
            grad_key = f'd{key}' # <-- ИСПРАВЛЕНИЕ
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[grad_key]
            params[key] += self.v[key]

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        if not self.m:
            for key in params.keys():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        for key in params.keys():
            grad_key = f'd{key}' # <-- ИСПРАВЛЕНИЕ
            g = grads[grad_key]
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g**2)
            
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# ==========================================
# 3. КЛАСС НЕЙРОСЕТИ (MLP)
# ==========================================
class MLP:
    def __init__(self, layer_sizes, activation='relu', optimizer='adam', learning_rate=0.01, l2_reg=0.0):
        """
        :param layer_sizes: список размеров слоев, например [8, 16, 8, 1]
        :param activation: функция активации для скрытых слоев ('relu', 'tanh')
        :param optimizer: 'sgd', 'momentum', 'adam'
        :param learning_rate: скорость обучения
        :param l2_reg: коэффициент L2-регуляризации
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation_name = activation
        self.act, self.act_deriv = ACTIVATIONS[activation]
        self.l2_reg = l2_reg
        
        # Инициализация оптимизатора
        if optimizer == 'sgd':
            self.optimizer = SGD(learning_rate)
        elif optimizer == 'momentum':
            self.optimizer = Momentum(learning_rate)
        elif optimizer == 'adam':
            self.optimizer = Adam(learning_rate)
        else:
            raise ValueError("Неизвестный оптимизатор. Используйте 'sgd', 'momentum' или 'adam'")

        self.params = {}
        self._init_weights()
        
        # Для хранения промежуточных значений при прямом проходе (нужно для backprop)
        self.cache = {}

    def _init_weights(self):
        """
        Инициализация весов He (для ReLU) или Xavier (для Tanh).
        """
        np.random.seed(42) # Для воспроизводимости
        for l in range(1, self.num_layers + 1):
            fan_in = self.layer_sizes[l - 1]
            fan_out = self.layer_sizes[l]
            
            if self.activation_name == 'relu':
                std = np.sqrt(2.0 / fan_in) # He initialization
            else:
                std = np.sqrt(1.0 / fan_in) # Xavier initialization
                
            self.params[f'W{l}'] = np.random.randn(fan_in, fan_out) * std
            self.params[f'b{l}'] = np.zeros((1, fan_out))

    def forward(self, X):
        """
        Прямой проход. Возвращает предсказания.
        """
        self.cache['A0'] = X
        A = X
        
        for l in range(1, self.num_layers + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            
            Z = np.dot(A, W) + b
            self.cache[f'Z{l}'] = Z
            
            if l == self.num_layers:
                # Выходной слой всегда линейный для регрессии
                A = linear(Z)
            else:
                A = self.act(Z)
                
            self.cache[f'A{l}'] = A
            
        return A

    def compute_loss(self, y_true, y_pred):
        """
        Вычисление MSE + L2 регуляризация.
        """
        m = y_true.shape[0]
        mse = np.mean((y_true - y_pred)**2)
        
        # L2 регуляризация (не применяем к bias)
        l2_loss = 0
        if self.l2_reg > 0:
            for l in range(1, self.num_layers + 1):
                l2_loss += np.sum(self.params[f'W{l}']**2)
            l2_loss = (self.l2_reg / (2 * m)) * l2_loss
            
        return mse + l2_loss

    def backward(self, y_true, y_pred):
        """
        Обратное распространение ошибки (Backpropagation).
        """
        m = y_true.shape[0]
        grads = {}
        
        # Производная MSE по выходу (выходной слой линейный)
        # dL/dA_L = 2 * (y_pred - y_true) / m
        dA = 2 * (y_pred - y_true) / m
        
        for l in reversed(range(1, self.num_layers + 1)):
            # Для выходного слоя производная линейной функции = 1
            if l == self.num_layers:
                dZ = dA
            else:
                dZ = dA * self.act_deriv(self.cache[f'Z{l}'])
                
            A_prev = self.cache[f'A{l-1}']
            
            # Градиенты по весам и смещениям
            grads[f'dW{l}'] = np.dot(A_prev.T, dZ)
            grads[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True)
            
            # L2 регуляризация для весов
            if self.l2_reg > 0:
                grads[f'dW{l}'] += (self.l2_reg / m) * self.params[f'W{l}']
            
            # Градиент для предыдущего слоя (если это не первый слой)
            if l > 1:
                dA = np.dot(dZ, self.params[f'W{l}'].T)
                
        return grads

    def train_step(self, X_batch, y_batch):
        """
        Один шаг обучения (Forward + Backward + Optimizer Step).
        """
        y_pred = self.forward(X_batch)
        loss = self.compute_loss(y_batch, y_pred)
        grads = self.backward(y_batch, y_pred)
        self.optimizer.update(self.params, grads)
        return loss

    def predict(self, X):
        return self.forward(X)

    def save_weights(self, path):
        """
        Сохранение весов модели в файл .npz
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **self.params)
        print(f"Модель сохранена в {path}")

    def load_weights(self, path):
        """
        Загрузка весов модели из файла .npz
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден.")
        with np.load(path) as data:
            for key in self.params.keys():
                self.params[key] = data[key]
        print(f"Веса успешно загружены из {path}")
