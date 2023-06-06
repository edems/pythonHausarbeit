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

        copy_of_function_data = df.copy()
        copy_of_function_data.columns = [name.capitalize() + table_name for name in copy_of_function_data.columns]
        copy_of_function_data.set_index(copy_of_function_data.columns[0], inplace=True)

        copy_of_function_data.to_sql(
            t_name,
            self.engine,
            if_exists="replace",
            index=True,
        )

    def clear_table(self):
        with self.engine.connect() as connection:   # Session erstellen
            Session = sessionmaker(bind=connection)
            session = Session()
            # Alle Tabellen löschen
            tables = self.inspector.get_table_names()
            for table_name in tables:
                drop_table_stmt = text(f"DROP TABLE {table_name}")
                connection.execute(drop_table_stmt)
            session.close()

    def write_all_table(self):
        with self.engine.connect() as connection:
            # Session erstellen
            Session = sessionmaker(bind=connection)
            session = Session()

            # Alle Tabellen löschen
            tables = self.inspector.get_table_names()

            for table in tables:
                print("Meine neue Table")
                print(table)

                # result = session.query(MeinetestTabelle).all()
                columns = self.inspector.get_columns(table)
                # Tabellenstruktur ausgeben
                print(f"Table: {table}")
                print("Columns:")
                for column in columns:
                    print(column['name'], column['type'])
                # Inhalt der Tabelle abrufen
                result_proxy = session.execute(text(f"SELECT * FROM {table}"))
                results = result_proxy.fetchall()

                # Ergebnisse ausgeben
                for row in results:
                    print(row)
            session.close()
    def  match_tosql(self, bestm, t_name, table_name):
        copy_of_function_data = bestm.copy()
        copy_of_function_data.columns = [name.capitalize() + table_name for name in copy_of_function_data.columns]
        copy_of_function_data.set_index(copy_of_function_data.columns[0], inplace=True)

        copy_of_function_data.to_sql(
            t_name,
            self.engine,
            if_exists="replace",
            index=True,
        )

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
    # pass
    def __init__(self, filename):
        DataBasis.__init__(self)
        self.data_frame = pd.read_csv(filename)
        self.abbildliste = []
        for datensatz in self.data_frame:
            if 'x' != datensatz:
                x = self.data_frame['x'].values
                y = self.data_frame[datensatz].values
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


class IdealData(DataBasis):
    def __init__(self, filename):
        DataBasis.__init__(self)
        self.data_frame = pd.read_csv(filename)
        self.abbildliste = []
        for datensatz in self.data_frame:
            if 'x' != datensatz:
                x = self.data_frame['x'].values
                y = self.data_frame[datensatz].values
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


class TestData(DataBasis):
    def __init__(self, filename):
        DataBasis.__init__(self)
        self.data_frame = pd.read_csv(filename)
        self.abbildliste = []
        for datensatz in self.data_frame:
            if 'x' != datensatz:
                x = self.data_frame['x'].values
                y = self.data_frame[datensatz].values
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

class BestFit(DataBasis):
    def __init__(self, train_df):
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

    def add_data_to_bestfit(self, abbild):
        self.abbildliste_bestfit.append(Abbild(abbild.x, abbild.y, abbild.name))

    def fill_bestfit_df(self):
        for ideal_function in self.abbildliste_bestfit:
            self.data_frame_bestfit = pd.concat([self.data_frame_bestfit, pd.DataFrame({'x': ideal_function.x,})]).drop_duplicates(subset='x', keep='first')
            self.data_frame_bestfit = pd.concat([self.data_frame_bestfit, pd.DataFrame({ideal_function.name: ideal_function.y})], axis=1)


class CheckFit(DataBasis):
    def __init__(self, test_data, best_fit):
        DataBasis.__init__(self)
        self.data_frame_test_data = test_data.data_frame
        self.data_frame_bestfit = best_fit.data_frame_bestfit
        self.data_frame_passendematch = pd.DataFrame()

        self.abbildliste_bestfit = best_fit.abbildliste_bestfit
        self.abbildliste_testdata = []
        self.abbildliste_passendepunkte = []
        self.distance = None
    def add_match_punkt(self, abbild):
        self.abbildliste_passendepunkte.append(abbild)
    def fill_match(self):
        for ideal_function in self.abbildliste_passendepunkte:
            tt = pd.DataFrame([[ideal_function.x, ideal_function.y, ideal_function.name, ideal_function.distance]] , columns=['X Test Data', 'Y Test Data', 'Kategorie', 'Distance'])
            self.data_frame_passendematch = pd.concat([self.data_frame_passendematch, tt])


class QuadraticFitting:
    def __init__(self):
        pass

    @staticmethod
    def fit2(train_data, ideal_functions):
        best_fits = BestFit(train_data.data_frame)
        for train_data in train_data:
            best_fit = None
            min_sum_squared_diff = float('inf')
            train_x = train_data.x
            train_y = train_data.y
            for ideal_function in ideal_functions:
                ideal_x = ideal_function.x
                ideal_y = ideal_function.y
                A = np.vstack([ideal_x, np.ones(len(ideal_x))]).T
                m, c = np.linalg.lstsq(A, ideal_y, rcond=None)[0]
                ideal_y_fit = m * train_x + c
                sum_squared_diff = np.sum(np.square(train_y - ideal_y_fit))

                if sum_squared_diff < min_sum_squared_diff:
                    min_sum_squared_diff = sum_squared_diff
                    best_fit = ideal_function
                    # print(j)
            best_fits.add_data_to_bestfit(best_fit)
        best_fits.fill_bestfit_df()
        return best_fits

    @staticmethod
    def show_aufgeins(best_fit):
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

    # @staticmethod
    # def show_aufgeins(best_fit):
    #     i = 0
    #     for train_data in best_fit.abbildliste_traindata:
    #         p = figure(title=f'Best Fit ', x_axis_label='x', y_axis_label='y')
    #         train_x = train_data.x
    #         train_y = train_data.y
    #
    #         p.scatter(train_x, train_y, legend_label='Train Data', color='blue')
    #         x = best_fit.abbildliste_bestfit[i].x
    #         y = best_fit.abbildliste_bestfit[i].y
    #         p.line(x, y, legend_label=best_fit.abbildliste_bestfit[i].name, color='red')
    #         output_file(f'best_fit_{i + 1}.html')
    #         show(p)
    #         i += 1
    #         p = None

    @staticmethod
    def locate_y_based_on_x(x, ideal_function):
        search_key = ideal_function["x"] == x
        return ideal_function.loc[search_key].iat[0, 1]

    @staticmethod
    def show_aufgzwei(bestm):
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
            mx = pd.DataFrame({'x': bestm.data_frame_bestfit["x"], 'y': bestm.data_frame_bestfit[figure_title]})
            figure_data["figure"].scatter(train_data.x, train_data.y, fill_color="red", legend_label="Test point",
                                          size=8)
            figure_data["figure"].line(mx["x"], mx["y"], legend_label=figure_title, color='blue')

        for figure_data in figures.values():
            output_file(figure_data["html"])
            show(figure_data["figure"])
        # p = figure(title="y36", x_axis_label='x', y_axis_label='y')
        # p2 = figure(title="y11", x_axis_label='x', y_axis_label='y')
        # p3 = figure(title="y2", x_axis_label='x', y_axis_label='y')
        # p4 = figure(title="y33", x_axis_label='x', y_axis_label='y')
        # for train_data in bestm.abbildliste_passendepunkte:
        #     if (train_data.name == "y36"):
        #         mx = pd.DataFrame(
        #             {'x': bestm.data_frame_bestfit["x"], 'y': bestm.data_frame_bestfit[train_data.name]})
        #         p.scatter(train_data.x, train_data.y, fill_color="red", legend_label="Test point", size=8)
        #         p.line(mx["x"], mx["y"], legend_label=train_data.name, color='blue')
        #     if (train_data.name == "y11"):
        #         mx2 = pd.DataFrame(
        #             {'x': bestm.data_frame_bestfit["x"], 'y': bestm.data_frame_bestfit[train_data.name]})
        #         p2.scatter(train_data.x, train_data.y, fill_color="red", legend_label="Test point", size=8)
        #         p2.line(mx2["x"], mx2["y"], legend_label=train_data.name, color='blue')
        #     if (train_data.name == "y2"):
        #         mx3 = pd.DataFrame(
        #             {'x': bestm.data_frame_bestfit["x"], 'y': bestm.data_frame_bestfit[train_data.name]})
        #         p3.scatter(train_data.x, train_data.y, fill_color="red", legend_label="Test point", size=8)
        #         p3.line(mx3["x"], mx3["y"], legend_label=train_data.name, color='blue')
        #     if (train_data.name == "y33"):
        #         mx4 = pd.DataFrame(
        #             {'x': bestm.data_frame_bestfit["x"], 'y': bestm.data_frame_bestfit[train_data.name]})
        #         p4.scatter(train_data.x, train_data.y, fill_color="red", legend_label="Test point", size=8)
        #         p4.line(mx4["x"], mx4["y"], legend_label=train_data.name, color='blue')
        #
        # output_file(f'best_1.html')
        # show(p)
        # output_file(f'best_2.html')
        # show(p2)
        # output_file(f'best_3.html')
        # show(p3)
        # output_file(f'best_4.html')
        # show(p4)

    @staticmethod
    def aufgbzwei(best_fit, test_data):
                # for x, y in zip(point_x, point_y):


        best_match = CheckFit(test_data, best_fit)
        for point in test_data:
            point_x = point.x
            point_y = point.y
            for x, y in zip(point_x, point_y):
                current_lowest_classification = None
                current_lowest_distance = None
                for ideal_function, train_f in zip(best_fit.abbildliste_bestfit, best_fit.abbildliste_traindata):
                    tx = pd.DataFrame({'x': best_fit.data_frame_train_data["x"], 'y': best_fit.data_frame_train_data[train_f.name]})
                    mx = pd.DataFrame({'x': best_fit.data_frame_bestfit["x"], 'y': best_fit.data_frame_bestfit[ideal_function.name]})

                    locate_y = QuadraticFitting.locate_y_based_on_x(x, mx)
                    distance = abs(locate_y - y)
                    matrixsubtraktion = tx - mx
                    matrixsubtraktion["y"] = matrixsubtraktion["y"].abs()
                    largest_deviation = math.sqrt(2) * max(matrixsubtraktion["y"])

                    if (abs(distance < largest_deviation)):
                        if ((current_lowest_classification == None) or (distance < current_lowest_distance)):
                            current_lowest_classification = ideal_function.name
                            current_lowest_distance = distance

                if (current_lowest_classification != None):
                    best_match.add_match_punkt(Abbild(x, y, current_lowest_classification, current_lowest_distance))
        best_match.fill_match()
        return best_match

train = TrainData('train.csv')
test = TestData('test.csv')
ideal = IdealData('ideal.csv')
# helper = MeineHelperKlasse()
# helper.clear_table()
# helper.df_into_sql(train.data_frame, "training", "training Data")
# helper.df_into_sql(ideal.data_frame, "ideal", "ideal Data")
ergebnis = QuadraticFitting.fit2(train, ideal)
QuadraticFitting.show_aufgeins(ergebnis)
# bestms = QuadraticFitting.aufgbzwei(ergebnis, test)
# helper.match_tosql(bestms.data_frame_passendematch, "matches", "Ergebnis Match")
# QuadraticFitting.show_aufgzwei(bestms)
# # helper.write_all_table()