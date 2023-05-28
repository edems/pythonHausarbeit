# from sqlalchemy import create_engine, Column, Integer, String, Float ,inspect
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
#
#
# username = 'adam'
# password = ''
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

class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TrainingData(Data):
    pass

class TestData(Data):
    pass

class IdealFunction(Data):
    pass # Implementierung der Vorhersagefunktion für die ideale Funktion


class QuadraticFitting:
    def __init__(self, train_data, ideal_functions):
        self.train_data = train_data
        self.ideal_functions = ideal_functions

    def fit2(self, test_data):
        best_fits = []
        test_deviations = []
        for train_data in self.train_data:
            best_fit = None
            min_sum_squared_diff = float('inf')
            train_x = train_data.x
            train_y = train_data.y
            print("neuer satz")
            j = 0
            for ideal_function in self.ideal_functions:
                j += 1
                ideal_x = ideal_function.x
                ideal_y = ideal_function.y
                if (train_y is None) or (ideal_y is None):
                    break
                else:
                    A = np.vstack([ideal_x, np.ones(len(ideal_x))]).T
                    m, c = np.linalg.lstsq(A, ideal_y, rcond=None)[0]
                    ideal_y_fit = m * train_x + c
                    sum_squared_diff = np.sum(np.square(train_y - ideal_y_fit))

                if sum_squared_diff < min_sum_squared_diff:
                    min_sum_squared_diff = sum_squared_diff
                    best_fit = ideal_function
                    print(j)
            best_fits.append(best_fit)

        return best_fits

# Funktion zum Laden der Daten aus einer CSV-Datei mit pandas
def load_data_from_csv(filename):
    data_frame = pd.read_csv(filename)
    x = data_frame['x'].values
    y = data_frame.drop('x', axis=1).values
    return x, y

# Daten aus den CSV-Dateien laden
train_x, train_y = load_data_from_csv('train.csv')
test_x, test_y = load_data_from_csv('test.csv')
ideal_x, ideal_y = load_data_from_csv('ideal.csv')

# Daten erstellen
train_data = [TrainingData(train_x, train_y[:, i]) for i in range(train_y.shape[1])]
test_data = [TestData(test_x, test_y[:, i]) for i in range(test_y.shape[1])]
ideal_functions2 = [IdealFunction(ideal_x, ideal_y[:, i]) for i in range(ideal_y.shape[1])]

quadratic_fitting2 = QuadraticFitting(train_data, ideal_functions2)
best_fits2 = quadratic_fitting2.fit2(ideal_functions2)

# Visualisierung der Daten und Anpassungen
for i, train_data in enumerate(train_data):
    p = figure(title=f'Best Fit {i+1}', x_axis_label='x', y_axis_label='y')
    p.scatter(train_data.x, train_data.y, legend_label='Train Data', color='blue')
    p.line(train_data.x, best_fits2[i].predict(train_data.x), legend_label='Best Fit2', color='red')
    output_file(f'best_fit_{i+1}.html')
    show(p)


