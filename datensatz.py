from sqlalchemy import create_engine, Column, Integer, String, Float, inspect, text, MetaData, Table
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
                for table_name in tables:
                    drop_table_stmt = text(f"DROP TABLE {table_name}")
                    connection.execute(drop_table_stmt)
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
    def __init__(self, filename):
        try:
            DataBasis.__init__(self)
            self.data_frame = pd.read_csv(filename)
            self.abbildliste = []
            for datensatz in self.data_frame:
                if 'x' != datensatz:
                    x = self.data_frame['x'].values
                    y = self.data_frame[datensatz].values
                    name = datensatz
                    self.abbildliste.append(Abbild(x, y, name))
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren von TrainData:", str(e))

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        try:
            if self.index >= len(self.abbildliste):
                raise StopIteration
            value = self.abbildliste[self.index]
            self.index += 1
            return value
        except StopIteration:
            raise
        except Exception as e:
            print("Fehler beim Abrufen des nächsten Elements:", str(e))

class IdealData(DataBasis):
    def __init__(self, filename):
        try:
            DataBasis.__init__(self)
            self.data_frame = pd.read_csv(filename)
            self.abbildliste = []
            for datensatz in self.data_frame:
                if 'x' != datensatz:
                    x = self.data_frame['x'].values
                    y = self.data_frame[datensatz].values
                    name = datensatz
                    self.abbildliste.append(Abbild(x, y, name))
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren der IdealData-Klasse:", str(e))


    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        try:
            if self.index >= len(self.abbildliste):
                raise StopIteration
            value = self.abbildliste[self.index]
            self.index += 1
            return value
        except StopIteration:
            raise
        except Exception as e:
            print("Fehler beim Iterieren über die IdealData-Klasse:", str(e))


class TestData(DataBasis):
    def __init__(self, filename):
        try:
            DataBasis.__init__(self)
            self.data_frame = pd.read_csv(filename)
            self.abbildliste = []
            for datensatz in self.data_frame:
                if 'x' != datensatz:
                    x = self.data_frame['x'].values
                    y = self.data_frame[datensatz].values
                    name = datensatz
                    self.abbildliste.append(Abbild(x, y, name))
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren der TestData-Klasse:", str(e))



    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.abbildliste):
            raise StopIteration
        value = self.abbildliste[self.index]
        self.index += 1
        return value

class BestFit(DataBasis):
    def __init__(self, train_df):
        try:
            DataBasis.__init__(self)
            self.data_frame_train_data = train_df
            self.data_frame_bestfit = pd.DataFrame()
            self.abbildliste_bestfit = []
            self.abbildliste_traindata = []
            for datensatz in self.data_frame_train_data:
                if 'x' != datensatz:
                    x = self.data_frame_train_data['x'].values
                    y = self.data_frame_train_data[datensatz].values
                    name = datensatz
                    self.abbildliste_traindata.append(Abbild(x, y, name))
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren der BestFit-Klasse:", str(e))

    def add_data_to_bestfit(self, abbild):
        self.abbildliste_bestfit.append(Abbild(abbild.x, abbild.y, abbild.name))

    def fill_bestfit_df(self):
        try:
            for ideal_function in self.abbildliste_bestfit:
                self.data_frame_bestfit = pd.concat(
                    [self.data_frame_bestfit, pd.DataFrame({'x': ideal_function.x})]).drop_duplicates(subset='x',
                                                                                                      keep='first')
                self.data_frame_bestfit = pd.concat(
                    [self.data_frame_bestfit, pd.DataFrame({ideal_function.name: ideal_function.y})], axis=1)
        except pd.errors.EmptyDataError:
            print("Der BestFit-Datenrahmen ist leer.")
        except pd.errors.ConcatenateError:
            print("Fehler beim Zusammenführen von DataFrames beim Füllen von data_frame_bestfit.")
        except pd.errors.DuplicateLabelError:
            print("Duplikate Labels wurden beim Füllen des BestFit-Datenrahmens gefunden.")
        except pd.errors.MergeError:
            print("Fehler beim Zusammenführen der DataFrames im BestFit-Datenrahmen.")
        except pd.errors.ParserError:
            print("Fehler beim Parsen der Daten im BestFit-Datenrahmen.")
        except pd.errors.EmptyDataError:
            print("Der BestFit-Datenrahmen ist leer.")
        except Exception as e:
            print("Ein allgemeiner Fehler ist beim Füllen des BestFit-Datenrahmens aufgetreten:", str(e))


class CheckFit(DataBasis):
    def __init__(self, test_data, best_fit):
        try:
            DataBasis.__init__(self)
            self.data_frame_test_data = test_data.data_frame
            self.data_frame_bestfit = best_fit.data_frame_bestfit
            self.data_frame_passendematch = pd.DataFrame()

            self.abbildliste_bestfit = best_fit.abbildliste_bestfit
            self.abbildliste_testdata = []
            self.abbildliste_passendepunkte = []
            self.distance = None
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren der BestFit-Klasse:", str(e))

    def add_match_punkt(self, abbild):
        try:
            self.abbildliste_passendepunkte.append(abbild)
        except Exception as e:
            print("Fehler beim Hinzufügen eines Übereinstimmungspunkts:", str(e))

    def fill_match(self):
        try:
            for ideal_function in self.abbildliste_passendepunkte:
                tt = pd.DataFrame([[ideal_function.x, ideal_function.y, ideal_function.name, ideal_function.distance]],
                                  columns=['X Test Data', 'Y Test Data', 'Kategorie', 'Distance'])
                self.data_frame_passendematch = pd.concat([self.data_frame_passendematch, tt])
        except pd.errors.ConcatenateError:
            print("Fehler beim Zusammenführen von DataFrames beim Füllen von data_frame_passendematch.")
        except Exception as e:
            print("Ein Fehler ist beim Füllen von data_frame_passendematch aufgetreten:", str(e))



class QuadraticFitting:
    def __init__(self):
        pass

    @staticmethod
    def fit2(train_data, ideal_functions):
        try:
            best_fits = BestFit(train_data.data_frame)
            for train_data in train_data:
                best_fit = None
                min_sum_squared_diff = float('inf')
                train_x = train_data.x
                train_y = train_data.y
                for ideal_function in ideal_functions:
                    ideal_x = ideal_function.x
                    ideal_y = ideal_function.y
                    try:
                        A = np.vstack([ideal_x, np.ones(len(ideal_x))]).T
                        m, c = np.linalg.lstsq(A, ideal_y, rcond=None)[0]
                        ideal_y_fit = m * train_x + c
                        sum_squared_diff = np.sum(np.square(train_y - ideal_y_fit))

                        if sum_squared_diff < min_sum_squared_diff:
                            min_sum_squared_diff = sum_squared_diff
                            best_fit = ideal_function
                    except np.linalg.LinAlgError as np_error:
                        # Fehler beim linearen Gleichungssystem lösen mit numpy
                        print("Fehler beim Lösen des linearen Gleichungssystems:", str(np_error))

                best_fits.add_data_to_bestfit(best_fit)
            best_fits.fill_bestfit_df()
            return best_fits

        except pd.errors.PandasError as pd_error:
            # Fehler bei der Verarbeitung von pandas-Datenstrukturen
            print("Fehler bei der Verarbeitung von pandas-Datenstrukturen:", str(pd_error))
            # Weitere Fehlerbehandlungsmaßnahmen durchführen oder den Fehler weiterreichen

        except Exception as e:
            # Allgemeiner Fehler
            print("Allgemeiner Fehler beim Ausführen der Methode fit2:", str(e))
            # Weitere Fehlerbehandlungsmaßnahmen durchführen oder den Fehler weiterreichen


    @staticmethod
    def show_aufgeins(best_fit):
        try:
            for i, train_data in enumerate(best_fit.abbildliste_traindata):
                p = figure(title=f'Best Fit ', x_axis_label='x', y_axis_label='y')
                train_x = train_data.x
                train_y = train_data.y
                p.scatter(train_x, train_y, legend_label='Train Data', color='blue')
                x = best_fit.abbildliste_bestfit[i].x
                y = best_fit.abbildliste_bestfit[i].y
                p.line(x, y, legend_label=best_fit.abbildliste_bestfit[i].name, color='red')
                output_file(f'best_fit_{i + 1}.html')
                show(p)
                p = None

        except Exception as e:
            # Allgemeiner Fehler
            print("Fehler beim Anzeigen der Grafik in der Aufgabe 1:", str(e))

    @staticmethod
    def locate_y_based_on_x(x, ideal_function):
        try:
            search_key = ideal_function["x"] == x
            return ideal_function.loc[search_key].iat[0, 1]
        except IndexError:
            # x-Wert nicht gefunden
            print("Der angegebene x-Wert wurde nicht gefunden.")
        except Exception as e:
            # Allgemeiner Fehler
            print("Fehler bei der Suche nach dem y-Wert:", str(e))


    @staticmethod
    def show_aufgzwei(bestm):
        try:
            figures = {}

            for train_data in bestm.abbildliste_passendepunkte:
                figure_title = train_data.name
                html_name = f'{figure_title}.html'

                if figure_title not in figures:
                    figures[figure_title] = {
                        "figure": figure(title=figure_title, x_axis_label='x', y_axis_label='y'),
                        "html": html_name
                    }

                figure_data = figures[figure_title]
                try:
                    mx = pd.DataFrame({'x': bestm.data_frame_bestfit["x"], 'y': bestm.data_frame_bestfit[figure_title]})
                    figure_data["figure"].scatter(train_data.x, train_data.y, fill_color="red",
                                                  legend_label="Test point", size=8)
                    figure_data["figure"].line(mx["x"], mx["y"], legend_label=figure_title, color='blue')
                except Exception as e:
                    print("Fehler beim Erstellen der Abbildungen:", str(e))

            for figure_data in figures.values():
                try:
                    output_file(figure_data["html"])
                    show(figure_data["figure"])
                except Exception as e:
                    print("Fehler beim Anzeigen der Abbildungen:", str(e))
        except Exception as e:
            print("Fehler bei der Ausführung von 'show_aufgzwei':", str(e))

    @staticmethod
    def aufgbzwei(best_fit, test_data):
        try:
            best_match = CheckFit(test_data, best_fit)
            for point in test_data:
                point_x = point.x
                point_y = point.y
                for x, y in zip(point_x, point_y):
                    current_lowest_classification = None
                    current_lowest_distance = None
                    for ideal_function, train_f in zip(best_fit.abbildliste_bestfit, best_fit.abbildliste_traindata):
                        try:
                            tx = pd.DataFrame({'x': best_fit.data_frame_train_data["x"],
                                               'y': best_fit.data_frame_train_data[train_f.name]})
                            mx = pd.DataFrame({'x': best_fit.data_frame_bestfit["x"],
                                               'y': best_fit.data_frame_bestfit[ideal_function.name]})

                            locate_y = QuadraticFitting.locate_y_based_on_x(x, mx)
                            distance = abs(locate_y - y)
                            matrixsubtraktion = tx - mx
                            matrixsubtraktion["y"] = matrixsubtraktion["y"].abs()
                            largest_deviation = math.sqrt(2) * max(matrixsubtraktion["y"])

                            if (abs(distance < largest_deviation)):
                                if ((current_lowest_classification == None) or (distance < current_lowest_distance)):
                                    current_lowest_classification = ideal_function.name
                                    current_lowest_distance = distance
                        except Exception as e:
                            print("Fehler bei der Berechnung:", str(e))

                    if (current_lowest_classification != None):
                        best_match.add_match_punkt(Abbild(x, y, current_lowest_classification, current_lowest_distance))
            best_match.fill_match()
            return best_match
        except Exception as e:
            print("Fehler bei der Ausführung von 'aufgbzwei':", str(e))


# train = TrainData('train.csv')
# test = TestData('test.csv')
# ideal = IdealData('ideal.csv')
# helper = MeineHelperKlasse()
# helper.clear_table()
# helper.df_into_sql(train.data_frame, "training", "training Data")
# helper.df_into_sql(ideal.data_frame, "ideal", "ideal Data")
# ergebnis = QuadraticFitting.fit2(train, ideal)
# QuadraticFitting.show_aufgeins(ergebnis)
# bestms = QuadraticFitting.aufgbzwei(ergebnis, test)
# helper.match_tosql(bestms.data_frame_passendematch, "matches", "Ergebnis Match")
# QuadraticFitting.show_aufgzwei(bestms)
# # helper.write_all_table()

class TestQuadraticFitting:
    def __init__(self):
        self.train_file = 'train.csv'
        self.test_file = 'test.csv'
        self.ideal_file = 'ideal.csv'
        self.helper = MeineHelperKlasse()

    def run(self):
        try:
            # Daten aus CSV-Dateien laden
            train = TrainData(self.train_file)
            test = TestData(self.test_file)
            ideal = IdealData(self.ideal_file)

            # Daten in die Datenbank schreiben
            self.helper.clear_table()
            self.helper.df_into_sql(train.data_frame, "training", "training Data")
            self.helper.df_into_sql(ideal.data_frame, "ideal", "ideal Data")

            # Quadratische Anpassung durchführen
            best_fit = QuadraticFitting.fit2(train, ideal)

            # Ergebnisse anzeigen
            QuadraticFitting.show_aufgeins(best_fit)

            # Anpassung an Testdaten durchführen
            best_match = QuadraticFitting.aufgbzwei(best_fit, test)

            # Ergebnisse in die Datenbank schreiben
            self.helper.match_tosql(best_match.data_frame_passendematch, "matches", "Ergebnis Match")

            # Zweite Ausgabe anzeigen
            QuadraticFitting.show_aufgzwei(best_match)

            # Alle Tabellen aus der Datenbank anzeigen
            self.helper.write_all_table()

        except FileNotFoundError:
            print("Eine oder mehrere der CSV-Dateien wurden nicht gefunden.")
        except PermissionError:
            print("Keine Berechtigung zum Lesen oder Schreiben der Dateien.")
        except ValueError:
            print("Ein Wertefehler ist aufgetreten.")
        except Exception as e:
            print("Fehler beim Ausführen des Tests:", str(e))


# Testklasse ausführen
test = TestQuadraticFitting()
test.run()
