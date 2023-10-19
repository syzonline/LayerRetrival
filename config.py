import json

def read_config():
    with open("config.json") as f:
        config = json.load(f)
    return config

config = read_config()
globals().update(config)