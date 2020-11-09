import yaml

class Config(object):
    def __init__(self, config_dict):
        self.config = config_dict

    def __getattr__(self, key):
        # Check if key exists
        if key in self.config:
            # Check if key has subkeys
            if type(self.config[key]) == dict:
                # Check which possibility is chosen
                for key2 in self.config[key]:
                    if key2 in self.config.values():
                        return self.config[key][key2]
            return self.config[key]
        else:
            return None
            # raise AttributeError(key)

    def __setitem__(self, key, value):
        self.config[key] = value


def get_config(path_to_config):

    with open(path_to_config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    return Config(config)