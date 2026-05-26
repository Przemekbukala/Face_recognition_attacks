from pathlib import Path
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

_config = None

def get_config():
    global _config
    if _config is None:
        with open(_REPO_ROOT / "config.yaml", "r") as f:
            _config = yaml.safe_load(f)
    return _config  

def get_attack_params(attack_name: str) -> dict:
    """
    Retrieve parameters for a given attack from the configuration.

    Parameters
    ----------
    attack_name : str
        Name of the attack.

    Returns
    -------
    dict
        Dictionary of parameters for the specified attack.
    """
    config = get_config()
    for attack in config.get("attacks"):
        if attack.get("name") == attack_name:
            return attack.get("parameters")
    return None

def get_attacks():
    config = get_config()
    return config.get("attacks", [])