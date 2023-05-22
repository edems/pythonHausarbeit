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
class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TrainingData(Data):
    pass

class TestData(Data):
    pass

class IdealFunction(Data):
    def predict(self, x):
        # Hier implementierst du die Vorhersagefunktion für die ideale Funktion
        pass

class QuadraticFitting:
    def __init__(self, train_data, ideal_functions):
        self.train_data = train_data
        self.ideal_functions = ideal_functions

    def fit(self):
        best_fits = []
        for train_data in self.train_data:
            best_fit = None
            min_sum_squared_diff = float('inf')
            for ideal_function in self.ideal_functions:
                ideal_y = ideal_function.predict(train_data.x)
                if (train_data.y is None) or (ideal_y is None):
                    break
                else:
                    sum_squared_diff = np.sum(np.square(train_data.y - ideal_y))
                if sum_squared_diff < min_sum_squared_diff:
                    min_sum_squared_diff = sum_squared_diff
                    best_fit = ideal_function
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

# Beispiel-Daten erstellen
train_data = [TrainingData(train_x, train_y[:, i]) for i in range(train_y.shape[1])]
test_data = [TestData(test_x, test_y[:, i]) for i in range(test_y.shape[1])]
ideal_functions = [IdealFunction(ideal_x, ideal_y[:, i]) for i in range(ideal_y.shape[1])]

# QuadraticFitting durchführen
quadratic_fitting = QuadraticFitting(train_data, ideal_functions)
best_fits = quadratic_fitting.fit()

# Ausgabe der besten Anpassungen
for i, fit in enumerate(best_fits):
    print(f"Best fit {i+1}: {fit}")

# Visualisierung der Daten und Anpassungen
for i, train_data in enumerate(train_data):
    # print("Das ist X" + train_data.x + " Das ist Y" + train_data.y + "Das ist best Fits" )
    # print("-------------------------------------------------------------------------------------------------------------------------------------------------")
    # print(" Das ist X")
    # print(train_data.x)
    # print(" Das ist Y")
    # print(train_data.y)
    plt.scatter(train_data.x, train_data.y, label='Train Data')
    # plt.plot(train_data.x, best_fits[i].predict(train_data.x), label='Best Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title(f'Best Fit {i+1}')
    plt.legend()

plt.show()
