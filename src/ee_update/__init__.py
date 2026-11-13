from .config import Config, REGIMES, regime
from .sim import Simulator
from .policies import make_policy, BASELINES, CONTROLLERS, ABLATIONS


def run_one(cfg: Config, policy_name: str, seed: int, trace: bool = False) -> dict:
    return Simulator(cfg, make_policy(policy_name), seed, trace=trace).run()
