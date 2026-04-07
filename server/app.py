"""
server/app.py
OpenEnv-core server with auto-generated UI.
"""

import sys
import os
import uvicorn

# Add project root to path so imports work when run as module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from env.environment import ISSEnvironment
from env.objects import Action, Observation

app = create_app(
    ISSEnvironment,
    Action,
    Observation,
    env_name="iss-safety-operations",
    max_concurrent_envs=1,
)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
