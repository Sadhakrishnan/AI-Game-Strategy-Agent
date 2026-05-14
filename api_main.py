from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from env.game_env import parallel_env
from rl.train import train
import os

app = FastAPI(title="Tactical AI Strategy API")

class TrainingConfig(BaseModel):
    timesteps: int = 100000

@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    background_tasks.add_task(train)
    return {"message": "Training started in background."}

@app.get("/metrics")
async def get_metrics():
    # In a real app, you'd pull this from TensorBoard logs or a database
    return {
        "status": "active",
        "latest_reward": 45.2,
        "win_rate": 0.62
    }

@app.get("/agents")
async def get_agents():
    return ["blue_0", "blue_1", "red_0", "red_1"]

@app.post("/reset")
async def reset_env():
    env = parallel_env()
    env.reset()
    return {"message": "Environment reset."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
