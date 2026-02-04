import os
from typing import List
import numpy as np
import django
from django.apps import apps
from django.db import models
import pandas as pd

import sys

class DBManager:

    def __init__(self, db_name: str):
        self._db_name = db_name
        self._settings_location = db_name + '.settings'
        # Dynamically find the correct project root
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", self._db_name))

        # Ensure the project root is in sys.path
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)

        # Set up Django settings
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", self._settings_location)
        django.setup()
        import InstrumentsApp.models as models

    def get_model_by_name(self, model_name):
        """Retrieve a Django model dynamically using its name."""
        try:
            return apps.get_model('InstrumentsApp', model_name)  # Replace with your app name
        except LookupError:
            raise ValueError(f"Model '{model_name}' not found in the app.")


    def queryset_to_dataframe(self, queryset):
        """Convert a Django queryset into a Pandas DataFrame."""
        if not queryset.exists():
            # print("Queryset is empty.")
            return pd.DataFrame()  # Return an empty DataFrame

        data = list(queryset.values())  # Convert queryset to a list of dictionaries
        df = pd.DataFrame(data)  # Convert to DataFrame
        return df


    def get_table_columns(self, model_name) -> List[str]:
        model = self.get_model_by_name(model_name)
        return [field.name for field in model._meta.get_fields() if field.name != 'id']


    def add_data_from_excel(self, model_name, file_path):
        table_columns_list = self.get_table_columns(model_name)

        # Read Excel file
        df = pd.read_excel(file_path)
        # df = df.melt(id_vars=['isin', 'ticker', 'hedged'], var_name='currency', value_name='weight')
        mismatching_columns = set(table_columns_list) != set(list(df.columns))
        if mismatching_columns:
            print(f"Mismatching columns: necessary columns are {', '.join(table_columns_list)} "
                  f"but given are {', '.join(list(df.columns))}")
            return
        df = df[table_columns_list]

        # Confirm before inserting data
        print(f"Data preview from {file_path}:\n")
        print(df.head())  # Show first few rows
        confirm = input("\nIs the data correct? (yes/no): ").strip().lower()

        if confirm.strip().lower() not in ["yes", "y"]:
            print("Data insertion canceled.")
            return

        # Insert data into database
        for _, row in df.iterrows():
            row_data = row.to_dict()
            self.add_data(model_name, **row_data)

        print("All data added successfully.")


    def add_data(self, model_name, **kwargs):
        try:
            """Dynamically add data to a given Django model."""
            model = self.get_model_by_name(model_name)
            model.validate_fields(kwargs)
            model.validate_choices(kwargs)
            instance = model(**kwargs)  # Create an instance of the model
            instance.save()  # Save to the database
            print(f"Data added successfully: {instance}")
        except Exception as e:
            print(e)


    def read_data(self, model_name, **lookup):
        """Read and return data from a given Django model using lookup filters."""
        try:
            model = self.get_model_by_name(model_name)
            queryset = model.objects.filter(**lookup)  # Fetch matching records

            if not queryset.exists():
                return None

            if queryset.count() == 1:
                # print(f"Data retrieved: {queryset.first()}")  # Single result
                return self.queryset_to_dataframe(queryset)

            # print(f"Multiple records found ({queryset.count()}):")
            # for instance in queryset:
            #     print(instance)
            return self.queryset_to_dataframe(queryset)  # Return all matching instances

        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None


    def delete_data(self, model_name, **lookup):
        """Delete data from a given Django model using lookup filters."""
        try:
            model = self.get_model_by_name(model_name)
            instances = model.objects.filter(**lookup)  # Find matching instances

            if not instances.exists():
                print("No matching records found.")
                return

            if instances.count() == 1:
                instance = instances.first()
                instance.delete()
                print(f"Data deleted successfully: {instance}")
            else:
                # print("Multiple records found:")
                # for idx, instance in enumerate(instances, start=1):
                #     print(f"{idx}. {instance}")

                confirm = input("Do you want to delete all? (yes/no): ").strip().lower()
                if confirm.strip().lower() in ["yes", "y"]:
                    count, _ = instances.delete()  # Deletes all instances
                    print(f"{count} records deleted successfully.")
                else:
                    print("Deletion canceled.")

        except Exception as e:
            print(f"Error: {e}")


    def get_tables(self):
        all_models = apps.get_models()
        tables_not_to_consider = ['django_admin_log', 'auth_permission', 'auth_group',
                                  'auth_user', 'django_content_type', 'django_session']
        for model in all_models:
            name = model._meta.db_table
            if name not in tables_not_to_consider:
                print(f"Table: {name}")  # Print table name


if __name__ == '__main__':
    db_manager = DBManager('C:\AFMachineLearning\Projects\Trading\FxdIncomeEtfAnalysis')
    # res = price_db_manager.add_data(model_name='Instruments', isin='LU2098180271', exchange='ETF Plus', trading_currency='EUR', fund_currency='JPY', ticker='JT13E')
    # res = price_db_manager.delete_data(model_name='Instruments', ticker='C73')
    res = db_manager.read_data(model_name='Brothers', cluster=187)
    # print(get_table_columns(model_name='YTMMapping'))
    # res = price_db_manager.add_data_from_excel(model_name='YTMMapping', file_path='C:\AFMachineLearning\Projects\Trading\MarketMonitorFI\InstrumentDbManager\ETF_FI_Instruments\excel_file.xlsx')
    print(res)