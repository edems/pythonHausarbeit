import unittest
from dataprocessing import TrainData, TestData, IdealData, QuadraticFitting
from sqlhelper import SQLHelperKlasse

class MyTestCase(unittest.TestCase):
    def test_something(self):


        try:
            self.train_file = 'train.csv'
            self.test_file = 'test.csv'
            self.ideal_file = 'ideal.csv'
            self.helper = SQLHelperKlasse()
            # Daten aus CSV-Dateien laden
            train = TrainData(self.train_file)
            test = TestData(self.test_file)
            ideal = IdealData(self.ideal_file)



        except FileNotFoundError:
            print("Eine oder mehrere der CSV-Dateien wurden nicht gefunden.")
        except PermissionError:
            print("Keine Berechtigung zum Lesen oder Schreiben der Dateien.")
        except ValueError:
            print("Ein Wertefehler ist aufgetreten.")
        except Exception as e:
            print("Fehler beim Ausführen des Tests:", str(e))
        # add assertion here
# class TestQuadraticFitting:
#     def __init__(self):
#
#
#     def run(self):
#

#
# # Testklasse ausführen
# test = TestQuadraticFitting()
# test.run()


if __name__ == '__main__':
    unittest.main()
