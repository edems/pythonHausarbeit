from sqlalchemy import create_engine, Column, Integer, String, Float, inspect, text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

class SQLHelperClass:
    def __init__(self):
        """
        Initializes an instance of the SQLHelperClass.
        """
        self.names = ""
        self.username = 'adam'
        self.password = 'KpCamSP0GZKrGGnan6uQ'
        self.host = 'hausarbeit.mysql.database.azure.com'
        self.database = 'hausarbeit'
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.username}:{self.password}@{self.host}/{self.database}", echo=True)
        self.inspector = inspect(self.engine)

    def write_dataframe_into_table(self, df, t_name, table_name):
        """
        Writes a DataFrame into an SQL table.

        Args:
            df (pandas.DataFrame): The DataFrame to be written.
            t_name (str): The name of the table in the database.
            table_name (str): The name of the table in the DataFrame.
        """
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
            print(f"Error writing {table_name} into the database:", str(e))

    def clear_tables(self):
        """
        Deletes all tables in the database.
        """
        try:
            with self.engine.connect() as connection:
                Session = sessionmaker(bind=connection)
                session = Session()
                tables = self.inspector.get_table_names()
                _ = [connection.execute(text(f"DROP TABLE {table_name}")) for table_name in tables]
                session.close()
        except Exception as e:
            print("Error clearing tables:", str(e))

    def write_all_table(self):
        """
        Reads all tables in the database and prints them.
        """
        try:
            with self.engine.connect() as connection:
                Session = sessionmaker(bind=connection)
                session = Session()
                tables = self.inspector.get_table_names()
                for table in tables:
                    print("My new table")
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
            print("Error reading tables:", str(e))

    def match_tosql(self, bestm, t_name, table_name):
        """
        Writes the match result into an SQL table.

        Args:
            bestm (pandas.DataFrame): The DataFrame with the match result.
            t_name (str): The name of the table in the database.
            table_name (str): The name of the table in the DataFrame.
        """
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
            print(f"Error writing the match result into the database:", str(e))
