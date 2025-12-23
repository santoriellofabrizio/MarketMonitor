import os
from functools import wraps
from zipfile import BadZipFile

import pandas as pd


def save_to_excel(sheet_name: str, file_name: str = ".cache/logging/excel_logging.xlsx"):
    """
    Decoratore che salva l'output_NAV di una funzione DataFrame in un file Excel su un foglio specificato.

    Args:
        file_name (str): Nome del file Excel dove salvare il DataFrame.
        sheet_name (str): Nome del foglio di lavoro su cui salvare i dati.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            while True:
                try:
                    # Esegui la funzione che restituisce un DataFrame
                    df = func(*args, **kwargs)

                    # Verifica che il risultato sia un DataFrame
                    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series):
                        print("La funzione decorata deve restituire un DataFrame")
                        return df

                    # Prova a scrivere nel file Excel
                    with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=True)
                    break  # Esci dal ciclo se l'operazione è riuscita

                except FileNotFoundError:
                    # Se il file non esiste, crea la cartella e un nuovo file
                    os.makedirs(os.path.dirname(file_name), exist_ok=True)  # Crea tutte le cartelle necessarie
                    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=True)

                    break  # Esci dal ciclo se il file è stato creato

                except PermissionError:
                    print("Il file Excel di logging è già aperto.")
                    input("\nChiudi il file e premi invio per continuare...\n")  # Attendi che l'utente chiuda il file

            return df

        return wrapper

    return decorator
