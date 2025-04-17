"""File for the load_config util function."""

from box import Box
from tomli import load


def load_config(config_path : str):
    """
    Function to load a configuration file.

    Args:
        config_path (str): the path to the configuration file.

    Returns:
        Box: the configuration loaded as a box.
    """
    with open(config_path, "rb") as f:
        config = load(f)

    return Box(config)
