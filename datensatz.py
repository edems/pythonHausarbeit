from sqlalchemy import create_engine, Column, Integer, String, Float, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook

output_notebook()


class MeineHelperKlasse:
    def __init__(self, filename):
        self.names = filename
        self.username = 'adam'
        self.password = 'KpCamSP0GZKrGGnan6uQ'
        self.host = 'hausarbeit.mysql.database.azure.com'
        self.database = 'hausarbeit'
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.username}:{self.password}@{self.host}/{self.database}", echo=True)
        self.inspector = inspect(self.engine)

    def df_into_sql(self, df, t_name, table_name):

        copy_of_function_data = df.copy()
        copy_of_function_data.columns = [name.capitalize() + table_name for name in copy_of_function_data.columns]
        copy_of_function_data.set_index(copy_of_function_data.columns[0], inplace=True)

        copy_of_function_data.to_sql(
            t_name,
            self.engine,
            if_exists="replace",
            index=True,
        )

    def clear_table(self):
        with self.engine.connect() as connection:
            # Session erstellen
            Session = sessionmaker(bind=connection)
            session = Session()
            # Alle Tabellen löschen
            tables = self.inspector.get_table_names()
            for table_name in tables:
                drop_table_stmt = text(f"DROP TABLE {table_name}")
                connection.execute(drop_table_stmt)
            session.close()

    def write_all_table(self):
        with self.engine.connect() as connection:
            # Session erstellen
            Session = sessionmaker(bind=connection)
            session = Session()

            # Alle Tabellen löschen
            tables = self.inspector.get_table_names()

            for table in tables:
                print("Meine neue Table")
                print(table)

                # result = session.query(MeinetestTabelle).all()
                columns = self.inspector.get_columns(table)
                # Tabellenstruktur ausgeben
                print(f"Table: {table}")
                print("Columns:")
                for column in columns:
                    print(column['name'], column['type'])
                # Inhalt der Tabelle abrufen
                # results = session.query(tables).all()
                result_proxy = session.execute(text(f"SELECT * FROM {table}"))
                results = result_proxy.fetchall()

                # Ergebnisse ausgeben
                for row in results:
                    print(row)
                # Tabelleninhalt ausgeben
                # print("Content:")
                # for row in results:
                #     print(row)
            session.close()


# class My:
#     def __int__(self, names):
#         self.names = names
# self.username = 'adam'
# self.password = 'KpCamSP0GZKrGGnan6uQ'
# self.host = 'hausarbeit.mysql.database.azure.com'
# self.database = 'hausarbeit'
# self.engine = create_engine(f"mysql+mysqlconnector://{self.username}:{self.password}@{self.host}/{self.database}", echo=True)
# self.inspector = inspect(self.engine)

# def df_into_sql(self, df, table_name):
#
#     copy_of_function_data = df.copy()
#     copy_of_function_data.columns = [name.capitalize() + table_name for name in copy_of_function_data.columns]
#     copy_of_function_data.set_index(copy_of_function_data.columns[0], inplace=True)
#
#     copy_of_function_data.to_sql(
#         df,
#         self.engine,
#         if_exists="replace",
#         index=True,
#     )
#
# def write_all_table(self):
#     with self.engine.connect() as connection:
#     # Session erstellen
#         Session = sessionmaker(bind=connection)
#         session = Session()
#
#         # Alle Tabellen löschen
#         tables = self.inspector.get_table_names()
#         # for table_name in tables:
#         #     drop_table_stmt = text(f"DROP TABLE {table_name}")
#         #     connection.execute(drop_table_stmt)
#         for table in tables:
#             print("Meine neue Table")
#             print(table)
#
#         # result = session.query(MeinetestTabelle).all()
#
#         # Ergebnis ausgeben
#         # for row in result:
#         #     print(row.id, row.column1, row.column2, row.column3)
#
#         # Verbindung schließen (automatisch durch das 'with'-Statement)
#         session.close()


# Basisklasse für die Tabellendeklaration
# Base = declarative_base()

# Tabelle deklarieren
# class MeinetestTabelle(Base):
#     __tablename__ = 'meinetesttable'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     column1 = Column(String(255))
#     column2 = Column(Integer)
#     column3 = Column(Float)

# Tabelle erstellen, wenn sie noch nicht existiert
# Base.metadata.create_all(bind=engine, checkfirst=True)

# Inspektor erstellen


# Verbindung aufbauen und Session verwenden
# with engine.connect() as connection:
#     # Session erstellen
#     Session = sessionmaker(bind=connection)
#     session = Session()
#
#     # Alle Tabellen löschen
#     tables = inspector.get_table_names()
#     for table_name in tables:
#         drop_table_stmt = text(f"DROP TABLE {table_name}")
#         connection.execute(drop_table_stmt)
#     for table in tables:
#         print("Meine neue Table")
#         print(table)
#
#     # result = session.query(MeinetestTabelle).all()
#
#     # Ergebnis ausgeben
#     # for row in result:
#     #     print(row.id, row.column1, row.column2, row.column3)
#
#     # Verbindung schließen (automatisch durch das 'with'-Statement)
#     session.close()

#
class DataBasis:
    def __init__(self):
        self.data_frame = pd.DataFrame
        self.name = ""
        self.abbildliste = []


class Abbild:
    def __init__(self, x, y, name, distance=None):
        self.x = x
        self.y = y
        self.name = name
        self.distance = distance


class TrainData(DataBasis):
    # pass
    def __init__(self, filename):
        DataBasis.__init__(self)
        self.data_frame = pd.read_csv(filename)
        self.abbildliste = []
        for datensatz in self.data_frame:
            if 'x' != datensatz:
                x = self.data_frame['x'].values
                y = self.data_frame[datensatz].values
                name = datensatz
                self.abbildliste.append(Abbild(x, y, name))

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.abbildliste):
            raise StopIteration
        value = self.abbildliste[self.index]
        self.index += 1
        return value


class IdealData(DataBasis):
    def __init__(self, filename):
        DataBasis.__init__(self)
        self.data_frame = pd.read_csv(filename)
        self.abbildliste = []
        for datensatz in self.data_frame:
            if 'x' != datensatz:
                x = self.data_frame['x'].values
                y = self.data_frame[datensatz].values
                name = datensatz
                self.abbildliste.append(Abbild(x, y, name))

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.abbildliste):
            raise StopIteration
        value = self.abbildliste[self.index]
        self.index += 1
        return value


class TestData(DataBasis):
    def __init__(self, filename):
        DataBasis.__init__(self)
        self.data_frame = pd.read_csv(filename)
        self.abbildliste = []
        for datensatz in self.data_frame:
            if 'x' != datensatz:
                x = self.data_frame['x'].values
                y = self.data_frame[datensatz].values
                name = datensatz
                self.abbildliste.append(Abbild(x, y, name))

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.abbildliste):
            raise StopIteration
        value = self.abbildliste[self.index]
        self.index += 1
        return value


class QuadraticFitting:
    def __init__(self, train_data, ideal_functions, test_data):
        self.train_data = train_data
        self.ideal_functions = ideal_functions
        self.test_data = test_data
        self.best_fits = []
        self.best_fits2 = []

    def fit2(self):

        for train_data in self.train_data:
            best_fit = None
            min_sum_squared_diff = float('inf')
            train_x = train_data.x
            train_y = train_data.y
            for ideal_function in self.ideal_functions:
                ideal_x = ideal_function.x
                ideal_y = ideal_function.y
                A = np.vstack([ideal_x, np.ones(len(ideal_x))]).T
                m, c = np.linalg.lstsq(A, ideal_y, rcond=None)[0]
                ideal_y_fit = m * train_x + c
                sum_squared_diff = np.sum(np.square(train_y - ideal_y_fit))

                if sum_squared_diff < min_sum_squared_diff:
                    min_sum_squared_diff = sum_squared_diff
                    best_fit = ideal_function
                    # print(j)
            self.best_fits.append(best_fit)

    def show_aufgeins(self):
        i = 0
        for train_data in train:
            p = figure(title=f'Best Fit ', x_axis_label='x', y_axis_label='y')
            train_x = train_data.x
            train_y = train_data.y

            p.scatter(train_x, train_y, legend_label='Train Data', color='blue')
            x = self.best_fits[i].x
            y = self.best_fits[i].y
            p.line(x, y, legend_label=self.best_fits[i].name, color='red')
            output_file(f'best_fit_{i + 1}.html')
            show(p)
            i += 1
            p = None

    def locate_y_based_on_x(self, x, ideal_function):
        search_key = ideal_function["x"] == x
        return ideal_function.loc[search_key].iat[0, 1]

    def show_aufgzwei(self):
        p = figure(title="y36", x_axis_label='x', y_axis_label='y')
        p2 = figure(title="y11", x_axis_label='x', y_axis_label='y')
        p3 = figure(title="y2", x_axis_label='x', y_axis_label='y')
        p4 = figure(title="y33", x_axis_label='x', y_axis_label='y')
        for train_data in self.best_fits2:
            if (train_data.name == "y36"):
                mx = pd.DataFrame(
                    {'x': self.ideal_functions.data_frame["x"], 'y': self.ideal_functions.data_frame[train_data.name]})
                p.scatter(train_data.x, train_data.y, fill_color="red", legend_label="Test point", size=8)
                p.line(mx["x"], mx["y"], legend_label=train_data.name, color='blue')
            if (train_data.name == "y11"):
                mx2 = pd.DataFrame(
                    {'x': self.ideal_functions.data_frame["x"], 'y': self.ideal_functions.data_frame[train_data.name]})
                p2.scatter(train_data.x, train_data.y, fill_color="red", legend_label="Test point", size=8)
                p2.line(mx2["x"], mx2["y"], legend_label=train_data.name, color='blue')
            if (train_data.name == "y2"):
                mx3 = pd.DataFrame(
                    {'x': self.ideal_functions.data_frame["x"], 'y': self.ideal_functions.data_frame[train_data.name]})
                p3.scatter(train_data.x, train_data.y, fill_color="red", legend_label="Test point", size=8)
                p3.line(mx3["x"], mx3["y"], legend_label=train_data.name, color='blue')
            if (train_data.name == "y33"):
                mx4 = pd.DataFrame(
                    {'x': self.ideal_functions.data_frame["x"], 'y': self.ideal_functions.data_frame[train_data.name]})
                p4.scatter(train_data.x, train_data.y, fill_color="red", legend_label="Test point", size=8)
                p4.line(mx4["x"], mx4["y"], legend_label=train_data.name, color='blue')

        output_file(f'best_1.html')
        show(p)
        output_file(f'best_2.html')
        show(p2)
        output_file(f'best_3.html')
        show(p3)
        output_file(f'best_4.html')
        show(p4)

    def aufgbzwei(self):
        z = 0
        for point in self.test_data:
            point_x = point.x
            point_y = point.y
            i = 0
            for x in point_x:
                current_lowest_classification = None
                current_lowest_distance = None
                j = 0
                for ideal_function in self.best_fits:
                    tx = pd.DataFrame({'x': self.train_data.abbildliste[j].x, 'y': self.train_data.abbildliste[j].y})
                    df = pd.DataFrame({'x': ideal_function.x, 'y': ideal_function.y})
                    mx = pd.DataFrame({'x': self.ideal_functions.data_frame["x"],
                                       'y': self.ideal_functions.data_frame[ideal_function.name]})

                    locate_y = self.locate_y_based_on_x(x, df)
                    distance = abs(locate_y - point_y[i])
                    distancess = tx - mx
                    distancess["y"] = distancess["y"].abs()
                    largest_deviation = math.sqrt(2) * max(distancess["y"])

                    if (abs(distance < largest_deviation)):
                        if ((current_lowest_classification == None) or (distance < current_lowest_distance)):
                            current_lowest_classification = ideal_function.name
                            current_lowest_distance = distance
                    j += 1

                if (current_lowest_classification != None):
                    print(str(z) + " X " + str(x) + " Y " + str(point_y[i]) + " klassif " + str(
                        current_lowest_classification) + " distance " + str(current_lowest_distance))
                    self.best_fits2.append(
                        Abbild(x, point_y[i], current_lowest_classification, current_lowest_distance))
                    z = z + 1
                i += 1


train = Data('train.csv')
test = Data('test.csv')
ideal = Data('ideal.csv')
helper = MeineHelperKlasse(4)
helper.clear_table()
helper.df_into_sql(train.data_frame, "training", "training Data")
helper.df_into_sql(ideal.data_frame, "ideal", "ideal Data")
helper.write_all_table()
# ergebnis = QuadraticFitting(train, ideal, test)
# ergebnis.fit2()
# ergebnis.show_aufgeins()
# ergebnis.aufgbzwei()
#
# ergebnis.show_aufgzwei()
print("fertig")
