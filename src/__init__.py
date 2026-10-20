import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

RULESET_VERSION = "3"
SCHEMA_VERSION = "1.0"


def load_environment(**kwargs):
    from luma_gym.environment import load_environment as load

    return load(**kwargs)
