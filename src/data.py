import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(csv_path, save_dir='data', fig_dir='results/figures', random_state=42):
    """
    Загружает данные, разделяет на train/val/test, нормализует (без утечек)
    и сохраняет массивы и параметры нормализации.
    """
    print("1. Загрузка данных...")
    df = pd.read_excel(csv_path)
    
    # Переименование столбцов (как в EDA)
    columns = [
        'Cement', 'Slag', 'Fly_Ash', 'Water', 'Superplasticizer', 
        'Coarse_Aggregate', 'Fine_Aggregate', 'Age', 'Strength'
    ]
    df.columns = columns

    # Отделяем признаки (X) от целевой переменной (y)
    X = df.drop('Strength', axis=1).values
    y = df['Strength'].values.reshape(-1, 1) # Делаем y двумерным (N, 1) для нейросети

    print(f"Исходные размеры: X={X.shape}, y={y.shape}")

    # 2. Разделение на Train (70%), Val (15%), Test (15%)
    print("\n2. Разделение на выборки (70/15/15)...")
    # Сначала отделяем тест (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )
    # Затем из оставшихся 85% отделяем валидацию (15% от изначального объема = 15/85 от остатка)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=random_state
    )

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")

    # 3. Нормализация (StandardScaler) - обучение ТОЛЬКО на train!
    print("\n3. Нормализация данных (fit только на train)...")
    
    # Для признаков
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Для целевой переменной (масштабирование y стабилизирует обучение нейросети)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # 4. Проверка отсутствия утечек (среднее на train_scaled должно быть ~0, std ~1)
    print(f"\nПроверка X_train после нормализации: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    print(f"Проверка X_val после нормализации:   mean={X_val_scaled.mean():.4f}, std={X_val_scaled.std():.4f}")

    # 5. Сохранение параметров нормализации (чтобы потом делать inverse_transform для y)
    print("\n4. Сохранение параметров нормализации...")
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(save_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))

    # 6. Сохранение подготовленных массивов в .npz
    print("Сохранение массивов в .npz...")
    np.savez(
        os.path.join(save_dir, 'prepared_data.npz'),
        X_train=X_train_scaled, y_train=y_train_scaled,
        X_val=X_val_scaled, y_val=y_val_scaled,
        X_test=X_test_scaled, y_test=y_test_scaled
    )
    
    # 7. Построение графиков до/после нормализации
    print("Построение графиков распределений до/после...")
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    feature_names = columns[:-1]
    
    for i in range(4): # Покажем первые 4 признака для примера
        sns.histplot(X_train[:, i], ax=axes[i*2], kde=True, color='blue', alpha=0.5)
        axes[i*2].set_title(f'{feature_names[i]} (До)')
        
        sns.histplot(X_train_scaled[:, i], ax=axes[i*2+1], kde=True, color='green', alpha=0.5)
        axes[i*2+1].set_title(f'{feature_names[i]} (После)')
        
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '06_normalization_effect.png'), dpi=300)
    plt.close()

    print("\nГотово! Данные подготовлены и сохранены.")
    return (X_train_scaled, y_train_scaled, 
            X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test_scaled)

def get_mini_batches(X, y, batch_size, shuffle=True, random_state=None):
    """
    Генератор мини-батчей для обучения нейросети.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
        
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]

if __name__ == "__main__":
    # Путь к CSV (подразумевается, что он в папке data)
    csv_path = 'data/Concrete_Data.xls' 
    
    if os.path.exists(csv_path):
        load_and_preprocess_data(csv_path)
    else:
        print(f"Файл {csv_path} не найден. Убедитесь, что датасет скачан.")
