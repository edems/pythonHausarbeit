import pandas as pd
import math
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
output_notebook()
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
            #lambda version
            # best_fits = BestFit(train_data.data_frame)
            # _ = [best_fits.add_data_to_bestfit(best_fit) for train_data in train_data
            #      for best_fit, min_sum_squared_diff in [(None, float('inf'))]
            #      for ideal_function in ideal_functions
            #      for ideal_x, ideal_y in [(ideal_function.x, ideal_function.y)]
            #      for A, m, c, ideal_y_fit, sum_squared_diff in [
            #          (np.vstack([ideal_x, np.ones(len(ideal_x))]).T,
            #           np.linalg.lstsq(A, ideal_y, rcond=None)[0],
            #           m * train_x + c,
            #           np.sum(np.square(train_y - ideal_y_fit)))]
            #      if sum_squared_diff < min_sum_squared_diff]
            # best_fits.fill_bestfit_df()

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

        except Exception as e:
            # Allgemeiner Fehler
            print("Allgemeiner Fehler beim Ausführen der Methode fit2:", str(e))


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
