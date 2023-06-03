# from sqlalchemy import create_engine, Column, Integer, String, Float ,inspect
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
#
#
# username = 'adam'
# password = 'KpCamSP0GZKrGGnan6uQ'
# host = 'hausarbeit.mysql.database.azure.com'
# database = 'hausarbeit'
# engine = create_engine(f"mysql+mysqlconnector://{username}:{password}@{host}/{database}", echo=True)
#
# # Basisklasse für die Tabellendeklaration
# Base = declarative_base()
#
# # Tabelle deklarieren
# class MeinetestTabelle(Base):
#     __tablename__ = 'meinetesttable'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     column1 = Column(String(255))
#     column2 = Column(Integer)
#     column3 = Column(Float)
#
# # Tabelle erstellen, wenn sie noch nicht existiert
# Base.metadata.create_all(bind=engine, checkfirst=True)
#
# # Inspektor erstellen
# inspector = inspect(engine)
#
#
# # Verbindung aufbauen und Session verwenden
# with engine.connect() as connection:
#     # Session erstellen
#     Session = sessionmaker(bind=connection)
#     session = Session()
#
#     # Weitere Operationen auf der Datenbank durchführen
#     # Tabellen in der Datenbank abrufen
#     tables = inspector.get_table_names()
#     for table in tables:
#         print("Meine neue Table")
#         print(table)
#
#     result = session.query(MeinetestTabelle).all()
#
#     # Ergebnis ausgeben
#     for row in result:
#         print(row.id, row.column1, row.column2, row.column3)
#
#     # Verbindung schließen (automatisch durch das 'with'-Statement)
#     session.close()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
output_notebook()


class Abbild:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name


class Data:
    def __init__(self, filename):
        data_frame = pd.read_csv(filename)
        self.abbildliste = []
        for datensatz in data_frame:
            if 'x' != datensatz:
                x = data_frame['x'].values
                y = data_frame[datensatz].values
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


# @property
#     def get_abbild(self):
#         return self.abbildliste


class QuadraticFitting:
    def __init__(self, train_data, ideal_functions):
        self.train_data = train_data
        self.ideal_functions = ideal_functions
        self.best_fits = []
        self.best_fits2 = []

    def fit2(self):

        for train_data in self.train_data:
            best_fit = None
            min_sum_squared_diff = float('inf')
            train_x = train_data.x
            train_y = train_data.y
            # print("neuer satz")
            # j = 0
            for ideal_function in self.ideal_functions:
                # j += 1
                ideal_x = ideal_function.x
                ideal_y = ideal_function.y
                # if (train_y is None) or (ideal_y is None):
                #     break
                # else:
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
            # p.line(train_x, train_y, legend_label='Train Data', color='blue')
            x = self.best_fits[i].x
            y = self.best_fits[i].y
            p.line(x, y, legend_label=self.best_fits[i].name, color='red')
            output_file(f'best_fit_{i+1}.html')
            show(p)
            i+=1
            p = None



train = Data('train.csv')
test = Data('test.csv')
ideal = Data('ideal.csv')
ergebnis = QuadraticFitting(train, ideal)
ergebnis.fit2()
ergebnis.show_aufgeins()
print("fertig")
