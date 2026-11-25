import os
import numpy as np
import json
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

# Импортируем наши модули
from train import train_mlp_gradient, train_mlp_ga
from mlp import MLP
import joblib

app = FastAPI(title="Concrete Strength Prediction API")

# Подключаем папку web для раздачи HTML/CSS/JS
# Убедимся, что папка существует
if not os.path.exists("../web"):
    os.makedirs("../web")
app.mount("/web", StaticFiles(directory="../web"), name="web")

# Хранилище статусов задач (в реальном проекте лучше использовать БД)
tasks_db = {}

# ==========================================
# МОДЕЛИ ЗАПРОСОВ (Pydantic)
# ==========================================
class TrainRequest(BaseModel):
    method: str  # "gradient" или "ga"
    use_library: bool = False
    layer_sizes: List[int] = [8, 16, 8, 1]
    activation: str = "relu"
    optimizer: str = "adam"
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    pop_size: int = 50
    generations: int = 50
    model_name: str = "my_model"

# ==========================================
# ЭНДПОИНТЫ API
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Отдает главную страницу"""
    with open("../web/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/train")
async def api_train(request: TrainRequest):
    """
    Запускает обучение модели и возвращает метрики.
    (В идеале это нужно делать через BackgroundTasks, чтобы не блокировать сервер, 
    но для курсовой мы сделаем синхронно для простоты).
    """
    print(f"Получен запрос на обучение: {request.method}")
    
    # Загружаем данные
    data = np.load('../data/prepared_data.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    model_path = f"../models/{request.model_name}.npz"
    
    try:
        if request.method == "gradient":
            mlp, history, time_spent = train_mlp_gradient(
                X_train, y_train, X_val, y_val,
                layer_sizes=request.layer_sizes,
                activation=request.activation,
                optimizer=request.optimizer,
                learning_rate=request.learning_rate,
                epochs=request.epochs,
                batch_size=request.batch_size,
                model_save_path=model_path
            )
        elif request.method == "ga":
            mlp, history, time_spent = train_mlp_ga(
                X_train, y_train, X_val, y_val,
                layer_sizes=request.layer_sizes,
                activation=request.activation,
                use_library=request.use_library,
                pop_size=request.pop_size,
                generations=request.generations,
                model_save_path=model_path
            )
        else:
            return {"status": "error", "message": "Неизвестный метод обучения"}

        # Считаем финальные метрики на валидации
        y_val_pred = mlp.predict(X_val)
        val_rmse = float(np.sqrt(np.mean((y_val - y_val_pred)**2)))
        
        # Сохраняем историю (для графиков)
        # В истории ГА ключи 'best' и 'mean', в градиенте 'train_loss' и 'val_loss'
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "val_rmse": val_rmse,
            "time_spent": time_spent,
            "history": history
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/models")
async def get_models():
    """Возвращает список доступных сохраненных моделей"""
    models_dir = "../models"
    if not os.path.exists(models_dir):
        return {"models": []}
    
    # Ищем файлы .npz
    models = [f.replace('.npz', '') for f in os.listdir(models_dir) if f.endswith('.npz')]
    return {"models": models}

@app.get("/api/predict")
async def get_predictions(model_name: str):
    """
    Возвращает реальные и предсказанные значения для тестовой выборки
    (для построения графика Scatter plot в GUI).
    """
    model_path = f"../models/{model_name}.npz"
    if not os.path.exists(model_path):
        return {"status": "error", "message": "Модель не найдена"}
        
    # Загружаем данные
    data = np.load('../data/prepared_data.npz')
    X_test, y_test = data['X_test'], data['y_test']
    
    # Восстанавливаем архитектуру (в реальном проекте её нужно сохранять в JSON, 
    # но для простоты мы пока предполагаем фиксированную [8, 16, 8, 1])
    # ВНИМАНИЕ: Чтобы график работал для любой архитектуры, нам нужно сохранять layer_sizes.
    # Пока используем дефолтную.
    mlp = MLP([8, 16, 8, 1], activation='relu')
    mlp.load_weights(model_path)
    
    # Делаем предсказания
    y_pred = mlp.predict(X_test)
    
    # Возвращаем в виде списков (для JSON)
    return {
        "y_true": y_test.flatten().tolist(),
        "y_pred": y_pred.flatten().tolist()
    }

# Запуск сервера (для локальной разработки)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)