from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR

def initialize_scheduler(config, optimizer, n_train_steps):
    # construct schedulers
    if config.get('scheduler') is None:
        return None
    elif config.get('scheduler')=='linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=n_train_steps,
            **config.get('scheduler_kwargs'))
        step_every_batch = True
        use_metric = False
    elif config.get('scheduler') == 'cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=n_train_steps,
            **config.get('scheduler_kwargs'))
        step_every_batch = True
        use_metric = False
    elif config.get('scheduler')=='ReduceLROnPlateau':
        assert config.get('scheduler_metric_name'), f'scheduler metric must be specified for {config.get("scheduler")}'
        scheduler = ReduceLROnPlateau(
            optimizer,
            **config.get('scheduler_kwargs'))
        step_every_batch = False
        use_metric = True
    elif config.get('scheduler') == 'StepLR':
        scheduler = StepLR(optimizer, **config.get('scheduler_kwargs'))
        step_every_batch = False
        use_metric = False
    elif config.get('scheduler') == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, **config.get('scheduler_kwargs'))
        step_every_batch = False
        use_metric = False
    else:
        raise ValueError('Scheduler not recognized.')
    # add a step_every_batch field
    scheduler.step_every_batch = step_every_batch
    scheduler.use_metric = use_metric
    return scheduler

def step_scheduler(scheduler, metric=None):
    if isinstance(scheduler, ReduceLROnPlateau):
        assert metric is not None
        scheduler.step(metric)
    else:
        scheduler.step()
