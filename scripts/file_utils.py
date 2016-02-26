

import os

def is_valid(key):
    return key.replace('-','').isalnum()

def load_key(filepath):
    assert os.path.exists(filepath), 'filepath: {} not found'.format(filepath)
    
    key = None
    with open(filepath, 'rb') as f:
        key = f.readline()
    if is_valid(key):
        return key
    else:
        raise ValueError('invalid key: {}'.format(key))