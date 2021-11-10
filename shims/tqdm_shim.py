from tqdm import tqdm
import os

class TqdmMock:
    n=0
    total=0
    def update(self, *args, **kwargs):
        pass
    def close(self, *args, **kwargs):
        pass

def tqdm_shim(*args, **kwargs):
    if os.environ['DISABLE_TQDM']:
        if len(args) > 0:
            return args[0]
        else:
            return TqdmMock()
    return tqdm(*args, **kwargs)