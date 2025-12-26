import os


def load_cache(cache_path):
    """Carica la cache da disco se esiste, altrimenti restituisce un dizionario vuoto."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            import pickle
            return pickle.load(f)
    return {}


def save_cache(cache, cache_path):
    """Salva la cache su disco."""
    with open(cache_path, "wb") as f:
        import pickle
        pickle.dump(cache, f)