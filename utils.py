import pickle

def save_to(obj, path):
    with open(path, 'wb') as f:
        return pickle.dump(obj, f)

def load_from(path):
    with open(path, 'rb') as f:
        return pickle.load(f)