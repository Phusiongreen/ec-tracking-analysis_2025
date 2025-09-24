import os
from pathlib import Path
from typing import Union

import yaml

"""
Global variable to save program call time.
"""


def read_parameters(parameter_file):
    """Reads in default parameters and replaces user defined parameters."""
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))

    param_base_file = Path(current_path).parent.joinpath("resources", "parameters.yml")

    with open(param_base_file, 'r') as yml_f:
        parameters = yaml.safe_load(yml_f)

    with open(parameter_file) as file:
        parameters_local = yaml.safe_load(file)

    # overwrite global parameters with local setting
    for key in parameters_local:
        parameters[key] = parameters_local[key]

    return parameters

def create_path_recursively(path: Union[str, Path]) -> bool:
    """Create a path. Creates missing parent folders.

    Args:
        path:
            Path to be created.

    Returns:
        True if successful.

    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    return True
