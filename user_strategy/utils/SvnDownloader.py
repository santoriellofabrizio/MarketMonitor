import os
import subprocess

FXDINCOMEETFDB_SVN_PATH = "https://svn.bansel.it/sys/AreaFinanza/AFMachineLearning/Projects/Trading/FxdIncomeEtfAnalysis/FxdIncomeEtfDB.db"

def download_fxdincomedb_from_svn(output_path):
    check_svn()

    folder_path = os.path.dirname(output_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    result = subprocess.run(["svn", "export", FXDINCOMEETFDB_SVN_PATH, output_path], capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"svn export failed from {FXDINCOMEETFDB_SVN_PATH} to {output_path}")

def check_svn():
    try:
        subprocess.run(["svn", "help"], capture_output=True, text=True)
    except Exception:
        print("'svn' is not recognized as an internal or external command, operable program or batch file.")
        raise
