"""Sst Hackathon Env environment server components."""

# Try absolute import first (for Docker/Production)
# Fall back to relative import (for Local Testing)
try:
    from server.SST_hackathon_env_environment import SstHackathonEnvironment
except ImportError:
    from .SST_hackathon_env_environment import SstHackathonEnvironment

__all__ = ["SstHackathonEnvironment"]