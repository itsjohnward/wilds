import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import torch

def initialize_transform(transform_name, config, dataset, is_training):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    For transforms that take in `(x, y)` and return `(transformed_x, transformed_y)`,
    set `do_transform_y` to True when initializing the WILDSSubset.    
    """
    if transform_name is None:
        return None
    elif transform_name=='bert':
        return initialize_bert_transform(config)
    elif transform_name=='image_base':
        return initialize_image_base_transform(config, dataset)
    elif transform_name=='image_resize_and_center_crop':
        return initialize_image_resize_and_center_crop_transform(config, dataset)
    elif transform_name=='poverty':
        return initialize_poverty_transform(is_training)
    elif transform_name=='rxrx1':
        return initialize_rxrx1_transform(is_training)
    else:
        raise ValueError(f"{transform_name} not recognized")

def initialize_bert_transform(config):
    assert 'bert' in config.get('model')
    assert config.get('max_token_length') is not None

    tokenizer = getBertTokenizer(config.get('model'))
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config.get('max_token_length'),
            return_tensors='pt')
        if config.get('model') == 'bert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        elif config.get('model') == 'distilbert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform

def getBertTokenizer(model):
    if model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f'Model: {model} not recognized.')

    return tokenizer

def initialize_image_base_transform(config, dataset):
    transform_steps = []
    if dataset.original_resolution is not None and min(dataset.original_resolution)!=max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))
    if config.get('target_resolution') is not None and config.get('dataset')!='fmow':
        transform_steps.append(transforms.Resize(config.get('target_resolution')))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform = transforms.Compose(transform_steps)
    return transform

def initialize_image_resize_and_center_crop_transform(config, dataset):
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    assert dataset.original_resolution is not None
    assert config.get('resize_scale') is not None
    scaled_resolution = tuple(int(res*config.get('resize_scale')) for res in dataset.original_resolution)
    if config.get('target_resolution') is not None:
        target_resolution = config.get('target_resolution')
    else:
        target_resolution = dataset.original_resolution
    transform = transforms.Compose([
        transforms.Resize(scaled_resolution),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def initialize_poverty_transform(is_training):
    if is_training:
        transforms_ls = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
            transforms.ToTensor()]
        rgb_transform = transforms.Compose(transforms_ls)

        def transform_rgb(img):
            # bgr to rgb and back to bgr
            img[:3] = rgb_transform(img[:3][[2,1,0]])[[2,1,0]]
            return img
        transform = transforms.Lambda(lambda x: transform_rgb(x))
        return transform
    else:
        return None

def initialize_rxrx1_transform(is_training):
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform
