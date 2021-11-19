import copy
from .algorithm import algorithm_defaults
from .model import model_defaults
from .scheduler import scheduler_defaults
from .data_loader import loader_defaults
from .datasets import dataset_defaults, split_defaults

DEFAULTS = {
    'dataset_kwargs': {},
    'download': False,
    'frac': 1.0,
    'version': None,
    'loader_kwargs': {},
    'eval_loader': 'standard',
    'model_kwargs': {},
    'loss_kwargs': {},
    'optimizer_kwargs': {},
    'scheduler_kwargs': {},
    'scheduler_metric_split': 'val',
    'evaluate_all_splits': True,
    'eval_splits': [],
    'eval_only': False,
    'eval_epoch': None,
    'device': 0,
    'seed': 0,
    'log_dir': './logs',
    'log_every': 50,
    'save_best': True,
    'save_last': True,
    'save_pred': True,
    'use_wandb': False,
    'progress_bar': False,
    'resume': False
}

def populate_defaults(config):
    """Populates hyperparameters with defaults implied by choices
    of other hyperparameters."""

    orig_config = copy.deepcopy(config)
    assert config.get('dataset') is not None, 'dataset must be specified'
    assert config.get('algorithm') is not None, 'algorithm must be specified'

    # implied defaults from choice of dataset
    config = populate_config(
        config,
        dataset_defaults[config.get('dataset')]
    )

    # implied defaults from choice of split
    if config.get('dataset') in split_defaults and config.get('split_scheme') in split_defaults[config.get('dataset')]:
        config = populate_config(
            config,
            split_defaults[config.get('dataset')][config.get('split_scheme')]
        )

    # implied defaults from choice of algorithm
    config = populate_config(
        config,
        algorithm_defaults[config.get('algorithm')]
    )

    # implied defaults from choice of loader
    config = populate_config(
        config,
        loader_defaults
    )
    # implied defaults from choice of model
    if config.get('model'): config = populate_config(
        config,
        model_defaults[config.get('model')],
    )

    # implied defaults from choice of scheduler
    if config.get('scheduler'): config = populate_config(
        config,
        scheduler_defaults[config.get('scheduler')]
    )

    # misc implied defaults
    if config.get('groupby_fields') is None:
        config['no_group_logging'] = True
    config['no_group_logging'] = bool(config.get('groupby_fields'))

    for key, val in DEFAULTS.items():
        if config.get(key) is None:
            config[key] = val

    # basic checks
    required_fields = [
        'split_scheme', 'train_loader', 'uniform_over_groups', 'batch_size', 'eval_loader', 'model', 'loss_function',
        'val_metric', 'val_metric_decreasing', 'n_epochs', 'optimizer', 'lr', 'weight_decay',
        ]
    for field in required_fields:
        assert config.get(field) is not None, f"Must manually specify {field} for this setup."

    # data loader validations
    # we only raise this error if the train_loader is standard, and
    # n_groups_per_batch or distinct_groups are
    # specified by the user (instead of populated as a default)
    if config.get('train_loader') == 'standard':
        if orig_config.get('n_groups_per_batch') is not None:
            raise ValueError("n_groups_per_batch cannot be specified if the data loader is 'standard'. Consider using a 'group' data loader instead.")
        if orig_config.get('distinct_groups') is not None:
            raise ValueError("distinct_groups cannot be specified if the data loader is 'standard'. Consider using a 'group' data loader instead.")

    return config

def populate_config(config, template: dict, force_compatibility=False):
    """Populates missing (key, val) pairs in config with (key, val) in template.
    Example usage: populate config with defaults
    Args:
        - config: namespace
        - template: dict
        - force_compatibility: option to raise errors if config.key != template[key]
    """
    if template is None:
        return config

    d_config = config
    for key, val in template.items():
        if not isinstance(val, dict): # config[key] expected to be a non-index-able
            if key not in d_config or d_config[key] is None:
                d_config[key] = val
            elif d_config[key] != val and force_compatibility:
                raise ValueError(f"Argument {key} must be set to {val}")

        else: # config[key] expected to be a kwarg dict
            for kwargs_key, kwargs_val in val.items():
                if key not in d_config:
                    d_config[key] = {}
                if kwargs_key not in d_config[key] or d_config[key].get(kwargs_key) is None:
                    d_config[key][kwargs_key] = kwargs_val
                elif d_config[key].get(kwargs_key) != kwargs_val and force_compatibility:
                    raise ValueError(f"Argument {key}[{kwargs_key}] must be set to {val}")
    return config
