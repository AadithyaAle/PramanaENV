import os
from openenv.core.env_server.http_server import create_app

# 1. Import your custom environment logic
try:
    from server.SST_hackathon_env_environment import SstHackathonEnvironment
except ImportError:
    from .SST_hackathon_env_environment import SstHackathonEnvironment

# 2. Import your data models
try:
    from models import Action, Observation
except ImportError:
    from ..models import Action, Observation

# 3. Create the FastAPI app
# We map your 'Action' and 'Observation' names to the ones the server expects
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)