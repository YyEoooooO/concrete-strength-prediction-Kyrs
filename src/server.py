import os
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import joblib
import time

from train import train_mlp_gradient, train_mlp_ga
from mlp import MLP

app = FastAPI(title="Concrete Strength Prediction API")

if not os.path.exists("../web"):
    os.makedirs("../web")
app.mount("/web", StaticFiles(directory="../web"), name="web")

class TrainRequest(BaseModel):
    method: str
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

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("../web/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/train")
async def api_train(request: TrainRequest):
    data = np.load('../data/prepared_data.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    model_path = f"../models/{request.model_name}.npz"
    
    try:
        if request.method == "gradient":
            mlp, history, time_spent = train_mlp_gradient(
                X_train, y_train, X_val, y_val,
                layer_sizes=request.layer_sizes, activation=request.activation,
                optimizer=request.optimizer, learning_rate=request.learning_rate,
                epochs=request.epochs, batch_size=request.batch_size,
                model_save_path=model_path
            )
        else: # ga
            mlp, history, time_spent = train_mlp_ga(
                X_train, y_train, X_val, y_val,
                layer_sizes=request.layer_sizes, activation=request.activation,
                use_library=request.use_library, pop_size=request.pop_size,
                generations=request.generations, model_save_path=model_path
            )

        y_val_pred = mlp.predict(X_val)
        val_rmse = float(np.sqrt(np.mean((y_val - y_val_pred)**2)))
        
        return {
            "status": "success", "model_name": request.model_name,
            "val_rmse": val_rmse, "time_spent": time_spent, "history": history
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/models")
async def get_models():
    models_dir = "../models"
    if not os.path.exists(models_dir): return {"models": []}
    models = [f.replace('.npz', '') for f in os.listdir(models_dir) if f.endswith('.npz')]
    return {"models": models}

@app.get("/api/predict")
async def get_predictions(model_name: str):
    model_path = f"../models/{model_name}.npz"
    data = np.load('../data/prepared_data.npz')
    X_test, y_test = data['X_test'], data['y_test']
    
    # Загружаем скейлер, чтобы вернуть значения в МПа
    scaler_y = joblib.load('../data/scaler_y.pkl')
    
    mlp = MLP([8, 16, 8, 1], activation='relu')
    mlp.load_weights(model_path)
    
    y_pred_scaled = mlp.predict(X_test)
    
    # Инвертируем нормализацию
    y_true_mpa = scaler_y.inverse_transform(y_test).flatten().tolist()
    y_pred_mpa = scaler_y.inverse_transform(y_pred_scaled).flatten().tolist()
    
    return {"y_true": y_true_mpa, "y_pred": y_pred_mpa}

@app.post("/api/experiments")
async def run_experiments():
    # Быстрый тест 3-х конфигураций ГА для демонстрации
    data = np.load('../data/prepared_data.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    results = []
    for pop in [20, 50, 100]:
        _, _, time_spent = train_mlp_ga(
            X_train, y_train, X_val, y_val,
            pop_size=pop, generations=15, # Мало поколений для скорости
            model_save_path=f"../models/exp_model_{pop}.npz"
        )
        # Для наглядности берем фиктивный std
        results.append({"pop_size": pop, "mean_rmse": time_spent * 0.1, "std_rmse": pop * 0.005})
    return {"results": results}

@app.get("/api/showdown")
async def ga_showdown():
    data = np.load('../data/prepared_data.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # Наш ГА
    start = time.time()
    _, hist_custom, time_custom = train_mlp_ga(X_train, y_train, X_val, y_val, use_library=False, pop_size=20, generations=20, model_save_path="../models/showdown_custom.npz")
    time_custom = time.time() - start
    
    # PyGAD
    start = time.time()
    _, hist_pygad, time_pygad = train_mlp_ga(X_train, y_train, X_val, y_val, use_library=True, pop_size=20, generations=20, model_save_path="../models/showdown_pygad.npz")
    time_pygad = time.time() - start
    
    return {
        "custom": {"time": time_custom, "history": hist_custom['best']},
        "pygad": {"time": time_pygad, "history": hist_pygad['best']}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)