import unittest
from dataprocessing import QuadraticFitting , TrainData, BestFits
import unittest
import pandas as pd
import numpy as np


class QuadraticFittingTestCase(unittest.TestCase):
    """
    Unittests für die QuadraticFitting-Klasse.
    """

    def setUp(self):
        # Initialisierung der Testdaten und -objekte
        self.train_data = pd.read_csv('train.csv')
        self.ideal_functions = pd.read_csv('ideal.csv')
        self.best_fit = pd.read_csv('test.csv')

    def tearDown(self):
        # Aufräumen nach jedem Testfall
        pass

    def test_search_y_at_x(self):
        # Testet die search_y_at_x-Methode der QuadraticFitting-Klasse.

        # Vorbereitung der Testdaten
        ideal_data = pd.DataFrame({'x': self.ideal_functions['x'], 'y': self.ideal_functions['y1']})
        x = -19.1
        expected_y = -0.2478342

        # Aufruf der zu testenden Methode
        y = QuadraticFitting.search_y_at_x(x, ideal_data)

        # Überprüfung des erwarteten Ergebnisses
        self.assertEqual(y, expected_y)

    def test_create_traindata_structure(self):
        # Testet die create_traindata_structure-Methode der QuadraticFitting-Klasse.

        # Vorbereitung der Testdaten
        traindata = self.train_data
        test_trainData = TrainData('train.csv')

        # Überprüfung, ob die erzeugte Datenstruktur korrekt ist
        self.assertTrue(test_trainData.data_frame.equals(traindata))

    def test_best_fits(self):
        # Testet die best_fits-Methode der QuadraticFitting-Klasse.

        # Vorbereitung der Testdaten und -objekte
        test_trainData = TrainData('train.csv')
        test_IdealData = TrainData('ideal.csv')
        test_best_fits = BestFits(test_trainData.data_frame)

        for train_data in test_trainData:
            fits = []

            # Extrahieren der x- und y-Werte aus dem Trainingsdatensatz
            train_x = train_data.x
            train_y = train_data.y

            for ideal_function in test_IdealData:
                ideal_x = ideal_function.x
                ideal_y = ideal_function.y

                try:
                    # Berechnung der besten Anpassung mit Hilfe von Numpy
                    m, c = np.linalg.lstsq(np.vstack([ideal_x, np.ones(len(ideal_x))]).T, ideal_y, rcond=None)[0]
                    sum_squared_diff = np.sum(np.square(train_y - (m * train_x + c)))
                    fits.append((ideal_function, sum_squared_diff))
                except np.linalg.LinAlgError as np_error:
                    # Fehler beim Lösen des linearen Gleichungssystems mit numpy
                    print("Fehler beim Lösen des linearen Gleichungssystems:", str(np_error))

            # Finden der besten Anpassung
            found_fit = min(fits, key=lambda fit: fit[1])[0] if fits else None
            test_best_fits.add_data_to_bestfit(found_fit)

        # Füllen der besten Anpassung in ein DataFrame
        test_best_fits.fill_bestfit_to_df()

        # Ausführen der zu testenden Methode
        best_fit = QuadraticFitting.best_fits(test_trainData, test_IdealData)

        # Überprüfung, ob die erzeugte beste Anpassung korrekt ist
        self.assertTrue(best_fit.data_frame_bestfit.equals(test_best_fits.data_frame_bestfit))


if __name__ == '__main__':
    # Führt die Unit-Tests aus
    unittest.main()
