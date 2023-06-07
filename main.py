from dataprocessing import TrainData, TestData, IdealData, QuadraticFitting
from sqlhelper import MeineHelperKlasse

if __name__ == '__main__':

    train = TrainData('train.csv')
    test = TestData('test.csv')
    ideal = IdealData('ideal.csv')
    helper = MeineHelperKlasse()
    helper.clear_table()
    helper.df_into_sql(train.data_frame, "training", "training Data")
    helper.df_into_sql(ideal.data_frame, "ideal", "ideal Data")
    ergebnis = QuadraticFitting.fit2(train, ideal)
    QuadraticFitting.show_aufgeins(ergebnis)
    bestms = QuadraticFitting.aufgbzwei(ergebnis, test)
    helper.match_tosql(bestms.data_frame_passendematch, "matches", "Ergebnis Match")
    QuadraticFitting.show_aufgzwei(bestms)
    helper.write_all_table()

