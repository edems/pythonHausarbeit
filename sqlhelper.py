from sqlalchemy import create_engine, Column, Integer, String, Float, inspect, text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

class MeineHelperKlasse:
    def __init__(self):
        self.names = ""
        self.username = 'adam'
        self.password = 'KpCamSP0GZKrGGnan6uQ'
        self.host = 'hausarbeit.mysql.database.azure.com'
        self.database = 'hausarbeit'
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.username}:{self.password}@{self.host}/{self.database}", echo=True)
        self.inspector = inspect(self.engine)

    def df_into_sql(self, df, t_name, table_name):
        try:
            copy_of_function_data = df.copy()
            copy_of_function_data.columns = [name.capitalize() + table_name for name in copy_of_function_data.columns]
            copy_of_function_data.set_index(copy_of_function_data.columns[0], inplace=True)

            copy_of_function_data.to_sql(
                t_name,
                self.engine,
                if_exists="replace",
                index=True,
            )
        except Exception as e:
            print(f"Fehler beim Schreiben von {table_name} in die Datenbank:", str(e))

    def clear_table(self):
        try:
            with self.engine.connect() as connection:
                Session = sessionmaker(bind=connection)
                session = Session()
                tables = self.inspector.get_table_names()
                # for table_name in tables:
                #     drop_table_stmt = text(f"DROP TABLE {table_name}")
                #     connection.execute(drop_table_stmt)
                #lambda funktion
                _ = [connection.execute(text(f"DROP TABLE {table_name}")) for table_name in tables]

                session.close()
        except Exception as e:
            print("Fehler beim Löschen der Tabellen:", str(e))

    def write_all_table(self):
        try:
            with self.engine.connect() as connection:
                Session = sessionmaker(bind=connection)
                session = Session()
                tables = self.inspector.get_table_names()
                for table in tables:
                    print("Meine neue Table")
                    print(table)
                    columns = self.inspector.get_columns(table)
                    print(f"Table: {table}")
                    print("Columns:")
                    for column in columns:
                        print(column['name'], column['type'])
                    result_proxy = session.execute(text(f"SELECT * FROM {table}"))
                    results = result_proxy.fetchall()

                    for row in results:
                        print(row)
                session.close()
        except Exception as e:
            print("Fehler beim Lesen der Tabellen:", str(e))

    def match_tosql(self, bestm, t_name, table_name):
        try:
            copy_of_function_data = bestm.copy()
            copy_of_function_data.columns = [name.capitalize() + table_name for name in copy_of_function_data.columns]
            copy_of_function_data.set_index(copy_of_function_data.columns[0], inplace=True)

            copy_of_function_data.to_sql(
                t_name,
                self.engine,
                if_exists="replace",
                index=True,
            )
        except Exception as e:
            print(f"Fehler beim Schreiben des Match-Ergebnisses in die Datenbank:", str(e))

