from dataprocessing import TrainData, TestData, IdealData, QuadraticFitting
from sqlhelper import SQLHelperKlasse

if __name__ == '__main__':
    # Initialisierung des TrainData-Objekts mit 'train.csv'
    train = TrainData('train.csv')
    # Initialisierung des IdealData-Objekts mit 'ideal.csv'
    ideal = IdealData('ideal.csv')
    # Initialisierung des SQLHelperKlasse-Objekts
    sql_helper = SQLHelperKlasse()

    # Löschen aller Tabellen in der Datenbank
    sql_helper.clear_table()
    # Schreiben des Trainingsdatensatzes in die Datenbank
    sql_helper.df_into_sql(train.data_frame, "training", "Training Data")
    # Schreiben des Idealdatensatzes in die Datenbank
    sql_helper.df_into_sql(ideal.data_frame, "ideal", "Ideal Data")
    # Durchführung der QuadraticFitting-Berechnung
    fits_result = QuadraticFitting.best_fits(train, ideal)
    # Anzeige des Ergebnisses von Aufgabe 1
    QuadraticFitting.show_task_one_result(fits_result)

    # Initialisierung des TestData-Objekts mit 'test.csv'
    test = TestData('test.csv')
    # Durchführung der Aufgabe 2 mit dem Ergebnis aus Aufgabe 1 und dem Testdatensatz
    bestma = QuadraticFitting.aufgbzwei(fits_result, test)
    # # Anzeige des Ergebnisses von Aufgabe 2
    QuadraticFitting.show_task_two_result(bestma)
    # # Schreiben des Match-Ergebnisses in die Datenbank mit passender idealer Funktion und die Abweichung
    sql_helper.match_tosql(bestma.data_frame_passendematch, "matches", "Result")
    # # Ausgabe aller Tabelleninhalte aus der Datenbank mithilfe von print()
    sql_helper.write_all_table()



