import warnings
import functools


def deprecated(reason):
    """
    Decoratore per contrassegnare funzioni o metodi come deprecati.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} è deprecato. {reason}",
                category=FutureWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# --- ESEMPIO DI UTILIZZO ---

@deprecated("Usa 'rtdata.get_subscription_manager()' al suo posto.")
def vecchia_funzione():
    print("Eseguo operazione legacy...")


if __name__ == "__main__":
    vecchia_funzione()
