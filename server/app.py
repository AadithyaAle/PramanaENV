import os
import sys

# This forces the bot to always find models.py, no matter where it runs from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app

# Absolute imports that will never crash
from server.SST_hackathon_env_environment import SstHackathonEnvironment
from models import Action, Observation

app = create_app(
    SstHackathonEnvironment,
    Action,
    Observation,
    env_name="SST_hackathon_env",
    max_concurrent_envs=1,
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

# The exact string the bot is searching for
if __name__ == '__main__':
    main()