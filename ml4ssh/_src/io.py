import pickle


def save_object(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    return None

def load_object(path):
    with open(path, "rb") as file:
        return pickle.load(file)