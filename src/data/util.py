import string


DATASET_REGISTRY = {}


def register_dataset(name):
    def add_to_registry(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return add_to_registry


def get_datamodule(config):
    ds_name = config.Data.dataset_name.value
    ds_name = ds_name.replace(string.punctuation, '')
    try:
        ds = DATASET_REGISTRY[ds_name]
    except KeyError:
        raise KeyError(f"Dataset {ds_name} was not found.")
    return ds.from_config(config)
