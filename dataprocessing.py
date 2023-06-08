import pandas as pd
import math
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool
output_notebook()
class DataBasis:
    """
    Eine Klasse zur Verwaltung von Daten wird für die weitere Vererbung benötigt.
    """

    def __init__(self):
        """
        Initialisiert die DataBasis-Klasse.
        """
        self.data_frame = pd.DataFrame
        self.name = ""
        self.abbildliste = []


class Abbild:
    """
   Eine Klasse, die für die Struktur der X Y Funktionen benötigt wird.
   """

    def __init__(self, x, y, name, distance=None):
        """
        Initialisiert ein Abbild-Objekt.

        Args:
            x: Der x-Wert des Abbilds.
            y: Der y-Wert des Abbilds.
            name: Der Name der Funktion.
            distance: Die Entfernung des Abbilds (optional).

        Returns:
            None
        """
        self.x = x
        self.y = y
        self.name = name
        self.distance = distance


class TrainData(DataBasis):
    """
    Eine Klasse zur Verwaltung von Trainingsdaten.
    Diese Klasse erbt von der DataBasis-Klasse.
    """

    def __init__(self, filename):
        """
        Initialisiert ein TrainData-Objekt mit den Daten aus einer CSV-Datei.

        Args:
            filename: Der Dateiname der CSV-Datei.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialisierung der DataBasis-Klasse
            self.data_frame = pd.read_csv(filename)  # Einlesen des CSV-Datei in ein DataFrame
            self.abbildliste = []  # Initialisierung einer leeren Abbildliste
            for datensatz in self.data_frame:
                if 'x' != datensatz:
                    x = self.data_frame['x'].values  # Extrahieren der x-Werte
                    y = self.data_frame[datensatz].values  # Extrahieren der y-Werte
                    name = datensatz  # Extrahieren des Namens des Datensatzes
                    self.abbildliste.append(Abbild(x, y, name))  # Hinzufügen eines Abbild-Objekts zur Abbildliste
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren von TrainData:", str(e))

    def __iter__(self):
        """
        Initialisiert den Iterator für die TrainData-Klasse.

        Returns:
            self: Der Iterator selbst.

        """
        self.index = 0
        return self

    def __next__(self):
        """
        Ruft das nächste Element des Iterators ab.

        Returns:
            value: Das nächste Element des Iterators.

        Raises:
            StopIteration: Wenn das Ende des Iterators erreicht ist.

        """
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
    """
    Eine Klasse zur Verwaltung von Idealdaten.
    Diese Klasse erbt von der DataBasis-Klasse.
    """

    def __init__(self, filename):
        """
        Initialisiert ein IdealData-Objekt mit den Daten aus einer CSV-Datei.

        Args:
            filename: Der Dateiname der CSV-Datei.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialisierung der DataBasis-Klasse
            self.data_frame = pd.read_csv(filename)  # Einlesen des CSV-Datei in ein DataFrame
            self.abbildliste = []  # Initialisierung einer leeren Abbildliste
            for datensatz in self.data_frame:
                if 'x' != datensatz:
                    x = self.data_frame['x'].values  # Extrahieren der x-Werte
                    y = self.data_frame[datensatz].values  # Extrahieren der y-Werte
                    name = datensatz  # Extrahieren des Namens des Datensatzes
                    self.abbildliste.append(Abbild(x, y, name))  # Hinzufügen eines Abbild-Objekts zur Abbildliste
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren der IdealData-Klasse:", str(e))


    def __iter__(self):
        """
        Initialisiert den Iterator für die IdealData-Klasse.

        Returns:
            self: Der Iterator selbst.

        """
        self.index = 0
        return self

    def __next__(self):
        """
        Ruft das nächste Element des Iterators ab.

        Returns:
            value: Das nächste Element des Iterators.

        Raises:
            StopIteration: Wenn das Ende des Iterators erreicht ist.

        """
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
    """
    Eine Klasse zur Verwaltung von Testdaten.
    Diese Klasse erbt von der DataBasis-Klasse.
    """

    def __init__(self, filename):
        """
        Initialisiert ein TestData-Objekt mit den Daten aus einer CSV-Datei.

        Args:
            filename: Der Dateiname der CSV-Datei.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialisierung der DataBasis-Klasse
            self.data_frame = pd.read_csv(filename)  # Einlesen des CSV-Datei in ein DataFrame
            self.abbildliste = []  # Initialisierung einer leeren Abbildliste
            for datensatz in self.data_frame:
                if 'x' != datensatz:
                    x = self.data_frame['x'].values  # Extrahieren der x-Werte
                    y = self.data_frame[datensatz].values  # Extrahieren der y-Werte
                    name = datensatz  # Extrahieren des Namens des Datensatzes
                    self.abbildliste.append(Abbild(x, y, name))  # Hinzufügen eines Abbild-Objekts zur Abbildliste
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren der TestData-Klasse:", str(e))


    def __iter__(self):
        """
        Initialisiert den Iterator für die TestData-Klasse.

        Returns:
            self: Der Iterator selbst.

        """
        self.index = 0
        return self

    def __next__(self):
        """
        Ruft das nächste Element des Iterators ab.

        Returns:
            value: Das nächste Element des Iterators.

        Raises:
            StopIteration: Wenn das Ende des Iterators erreicht ist.

        """
        if self.index >= len(self.abbildliste):
            raise StopIteration
        value = self.abbildliste[self.index]
        self.index += 1
        return value


class BestFits(DataBasis):
    """
    Eine Klasse zur Verwaltung der besten Fits (besten Passungen).
    Diese Klasse erbt von der DataBasis-Klasse.
    """

    def __init__(self, train_df):
        """
        Initialisiert ein BestFits-Objekt mit den Trainingsdaten.

        Args:
            train_df: Das DataFrame der Trainingsdaten.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialisierung der DataBasis-Klasse
            self.data_frame_train_data = train_df  # DataFrame der Trainingsdaten
            self.data_frame_bestfit = pd.DataFrame()  # DataFrame für die besten Fits
            self.abbildliste_bestfit = []  # Liste der besten Fits (Abbildungen)
            self.abbildliste_traindata = []  # Liste der Trainingsdaten (Abbildungen)
            for datensatz in self.data_frame_train_data:
                if 'x' != datensatz:
                    x = self.data_frame_train_data['x'].values  # Extrahieren der x-Werte
                    y = self.data_frame_train_data[datensatz].values  # Extrahieren der y-Werte
                    name = datensatz  # Extrahieren des Namens des Datensatzes
                    self.abbildliste_traindata.append(Abbild(x, y, name))  # Hinzufügen eines Abbild-Objekts zur Trainingsdaten-Liste
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren der BestFits-Klasse:", str(e))

    def add_data_to_bestfit(self, abbild):
        """
        Fügt einen Fit zur Liste der besten Fits hinzu.

        Args:
            abbild: Das Abbild-Objekt, das hinzugefügt werden soll.

        Returns:
            None
        """
        self.abbildliste_bestfit.append(Abbild(abbild.x, abbild.y, abbild.name))

    def fill_bestfit_to_df(self):
        """
        Füllt den BestFit-Datenrahmen mit den gefundenen Fits.

        Returns:
            None
        """
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
            print("Duplikate Labels wurden beim Füllen des BestFits-Datenrahmens gefunden.")
        except pd.errors.MergeError:
            print("Fehler beim Zusammenführen der DataFrames im BestFits-Datenrahmen.")
        except pd.errors.ParserError:
            print("Fehler beim Parsen der Daten im BestFits-Datenrahmen.")
        except pd.errors.EmptyDataError:
            print("Der BestFits-Datenrahmen ist leer.")
        except Exception as e:
            print("Ein allgemeiner Fehler ist beim Füllen des BestFits-Datenrahmens aufgetreten:", str(e))



class CheckMappingFit(DataBasis):
    """
    Eine Klasse zum Überprüfen der Zuordnung des Fits.
    Diese Klasse erbt von der DataBasis-Klasse.
    """

    def __init__(self, test_data, best_fit):
        """
        Initialisiert ein CheckMappingFit-Objekt mit Testdaten und dem besten Fit.

        Args:
            test_data: Das Testdaten-Objekt.
            best_fit: Das BestFits-Objekt.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialisierung der DataBasis-Klasse
            self.data_frame_test_data = test_data.data_frame  # DataFrame der Testdaten
            self.data_frame_bestfit = best_fit.data_frame_bestfit  # DataFrame des besten Fits
            self.data_frame_passendematch = pd.DataFrame()  # DataFrame für die passenden Matches

            self.abbildliste_bestfit = best_fit.abbildliste_bestfit  # Liste der besten Fits (Abbildungen)
            self.abbildliste_testdata = []  # Liste der Testdaten (Abbildungen)
            self.abbildliste_passendepunkte = []  # Liste der passenden Punkte (Abbildungen)
            self.distance = None  # Entfernung (Distance)
        except FileNotFoundError:
            print("Die angegebene Datei wurde nicht gefunden.")
        except Exception as e:
            print("Fehler beim Initialisieren der CheckMappingFit-Klasse:", str(e))

    def add_match_punkt(self, abbild):
        """
        Fügt einen passenden Punkt zur Liste der passenden Punkte hinzu.

        Args:
            abbild: Das Abbild-Objekt, das hinzugefügt werden soll.

        Returns:
            None
        """
        try:
            self.abbildliste_passendepunkte.append(abbild)
        except Exception as e:
            print("Fehler beim Hinzufügen eines Übereinstimmungspunkts:", str(e))

    def fill_match_to_df(self):
        """
        Füllt den passenden Match-Datenrahmen mit den passenden Punkten.

        Returns:
            None
        """
        try:
            for ideal_function in self.abbildliste_passendepunkte:
                tt = pd.DataFrame([[ideal_function.x, ideal_function.y, ideal_function.name, ideal_function.distance]],
                                  columns=['X (Test Data)', 'Y (Test Data)', ' Passende Ideale Funktion', ' (Abweichung)'])
                self.data_frame_passendematch = pd.concat([self.data_frame_passendematch, tt])
        except pd.errors.ConcatenateError:
            print("Fehler beim Zusammenführen von DataFrames beim Füllen von data_frame_passendematch.")
        except Exception as e:
            print("Ein Fehler ist beim Füllen von data_frame_passendematch aufgetreten:", str(e))




class QuadraticFitting:
    """
    Eine Klasse zur Durchführung des quadratischen Fits.
    """

    def __init__(self):
        """
        Initialisiert ein QuadraticFitting-Objekt.
        Ein Standardkonstruktor für die enthaltene @staticmethod

        Returns:
            None
        """
        pass

    @staticmethod
    def best_fits(train_data, ideal_functions):
        """
        Ermittelt die besten Fits für die Trainingsdaten und die idealen Funktionen.

        Args:
            train_data: Das Trainingsdaten-Objekt.
            ideal_functions: Eine Liste der idealen Funktionen.

        Returns:
            best_fits: Das BestFits-Objekt mit den besten Fits.
        """
        try:
            best_fits = BestFits(train_data.data_frame)

            for train_data in train_data:
                found_fit = None
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
                            found_fit = ideal_function
                    except np.linalg.LinAlgError as np_error:
                        # Fehler beim Lösen des linearen Gleichungssystems mit numpy
                        print("Fehler beim Lösen des linearen Gleichungssystems:", str(np_error))

                best_fits.add_data_to_bestfit(found_fit)

            best_fits.fill_bestfit_to_df()
            return best_fits

        except pd.errors.PandasError as pd_error:
            # Fehler bei der Verarbeitung von pandas-Datenstrukturen
            print("Fehler bei der Verarbeitung von pandas-Datenstrukturen:", str(pd_error))

        except Exception as e:
            # Allgemeiner Fehler
            print("Allgemeiner Fehler beim Ausführen der Methode bestfit:", str(e))

    @staticmethod
    def show_task_one_result(best_fit):
        """
        Zeigt die Ergebnisse für Aufgabe 1 an.

        Parameters:
            best_fit (BestFit): Die BestFit-Instanz mit den besten Abbildungen.

        Returns:
            None
        """
        try:
            for i, train_data in enumerate(best_fit.abbildliste_traindata):
                p = figure(title=f'Best Fits für {best_fit.abbildliste_bestfit[i].name}', x_axis_label='x',
                           y_axis_label='y', width=1200, height=900)
                train_x = train_data.x
                train_y = train_data.y

                # Scatter-Diagramm für Trainingsdaten erstellen
                p.scatter(train_x, train_y, legend_label='Train Data', color='blue', size=5)

                x = best_fit.abbildliste_bestfit[i].x
                y = best_fit.abbildliste_bestfit[i].y

                # Linie für ideale Funktion hinzufügen
                p.line(x, y, legend_label=f'Ideale Funktion {best_fit.abbildliste_bestfit[i].name}', color='red',
                       line_width=2)

                output_file(f'best_fit_{i + 1}.html')
                show(p)
                p = None
        except Exception as e:
            # Allgemeiner Fehler
            print("Fehler beim Anzeigen der Grafik in der Aufgabe 1:", str(e))

    @staticmethod
    def search_y_at_x(x, ideal_data):
        """
        Sucht den y-Wert für einen gegebenen x-Wert in den idealen Daten.

        Parameters:
            x (float): Der gesuchte x-Wert.
            ideal_data (DataFrame): Die idealen Daten.

        Returns:
            float: Der entsprechende y-Wert.
        """
        try:
            s_key = ideal_data["x"] == x
            return ideal_data.loc[s_key].iat[0, 1]
        except IndexError:
            # x-Wert nicht gefunden
            print("Der angegebene x-Wert wurde nicht gefunden.")
        except Exception as e:
            # Allgemeiner Fehler
            print("Fehler bei der Suche nach dem y-Wert:", str(e))

    @staticmethod
    def show_task_two_result(bestm):
        """
        Zeigt die Ergebnisse für Aufgabe 2 an.

        Parameters:
            bestm (BestM): Die BestM-Instanz mit den Ergebnissen.

        Returns:
            None
        """
        try:
            figures = {}  # Dictionary zur Speicherung der Abbildungen

            # Schleife über die passenden Punkte
            for train_data in bestm.abbildliste_passendepunkte:
                figure_title = train_data.name
                html_name = f'{figure_title}.html'

                if figure_title not in figures:
                    figures[figure_title] = {
                        "figure": figure(
                            title=str("Die Passende Test Punkte für die Ideale Funktion " + train_data.name),
                            x_axis_label='x', y_axis_label='y', width=1200, height=900),
                        "html": html_name
                    }

                figure_data = figures[figure_title]

                try:
                    # Erstellen des DataFrames für die ideale Funktion
                    ideale_func_df = pd.DataFrame(
                        {'x': bestm.data_frame_bestfit["x"], 'y': bestm.data_frame_bestfit[figure_title]})

                    # Scatter-Diagramm erstellen
                    scatter = figure_data["figure"].scatter(train_data.x, train_data.y, fill_color="red",
                                                            legend_label="Test point", size=8)

                    # Hover-Tool hinzufügen, um Informationen bei Hover anzuzeigen
                    hover = HoverTool(renderers=[scatter], tooltips=[('Abweichung', f'{train_data.distance}')],
                                      mode='mouse')
                    figure_data["figure"].add_tools(hover)

                    # Linie für die ideale Funktion hinzufügen
                    figure_data["figure"].line(ideale_func_df["x"], ideale_func_df["y"], legend_label=figure_title,
                                               color='blue')
                except Exception as e:
                    print("Fehler beim Erstellen der Abbildungen:", str(e))

            # Anzeigen der Abbildungen
            for figure_data in figures.values():
                try:
                    output_file(figure_data["html"])
                    show(figure_data["figure"])
                except Exception as e:
                    print("Fehler beim Anzeigen der Abbildungen:", str(e))
        except Exception as e:
            print("Fehler bei der Ausführung von 'show_task_two_result':", str(e))

    @staticmethod
    def aufgbzwei(best_fit, test_data):
        """
        Führt Aufgabe 2 aus, indem die besten Übereinstimmungen zwischen Testdatenpunkten und idealen Funktionen ermittelt werden.

        Parameters:
            best_fit (BestFit): Die BestFit-Instanz mit den besten Abbildungen.
            test_data (List[Point]): Die Liste der Testdatenpunkte.

        Returns:
            CheckMappingFit: Die CheckMappingFit-Instanz mit den besten Übereinstimmungen.
        """
        try:
            best_match = CheckMappingFit(test_data, best_fit)

            # Schleife über die Testdatenpunkte
            for point in test_data:
                point_x = point.x
                point_y = point.y
                # Schleife für die einzelbetrachtung der Testdatenpunkte
                for x, y in zip(point_x, point_y):
                    actual_match = None
                    actual_distance = None

                    # Schleife über die idealen Funktionen und zugehörige Trainingsdaten
                    for ideal_function, train_f in zip(best_fit.abbildliste_bestfit, best_fit.abbildliste_traindata):
                        try:
                            # Erstellen der DataFrames für die Trainings- und besten Fits-Daten
                            train_data_df = pd.DataFrame({'x': best_fit.data_frame_train_data["x"],
                                                          'y': best_fit.data_frame_train_data[train_f.name]})
                            best_fit_df = pd.DataFrame({'x': best_fit.data_frame_bestfit["x"],
                                                        'y': best_fit.data_frame_bestfit[ideal_function.name]})

                            # Suche nach dem y-Wert an der gegebenen x-Position
                            locate_y = QuadraticFitting.search_y_at_x(x, best_fit_df)

                            # Berechnung der Abweichungen
                            distance = abs(locate_y - y)
                            matrixsubtraktion = train_data_df - best_fit_df
                            matrixsubtraktion["y"] = matrixsubtraktion["y"].abs()
                            largest_divergence = math.sqrt(2) * max(matrixsubtraktion["y"])

                            # Überprüfung auf beste Übereinstimmung
                            if (abs(distance < largest_divergence)):
                                if ((actual_match == None) or (distance < actual_distance)):
                                    actual_match = ideal_function.name
                                    actual_distance = distance
                        except Exception as e:
                            print("Fehler bei der Berechnung:", str(e))

                    if (actual_match != None):
                        best_match.add_match_punkt(Abbild(x, y, actual_match, actual_distance))

            best_match.fill_match_to_df()
            return best_match
        except Exception as e:
            print("Fehler bei der Ausführung von 'aufgbzwei':", str(e))


