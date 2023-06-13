import pandas as pd
import math
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.models import HoverTool

output_notebook()


class DataErrorException(Exception):
    """
    An exception that occurs when an error is encountered in the data.
    """

    def __init__(self, data):
        """
        Initializes a new instance of the DataErrorException.

        Args:
            data: The erroneous data.
        """
        self.data = data

    def __str__(self):
        """
        Returns a string representing the exception along with the data.
        """
        dt = pd.DataFrame(self.data)
        return f"An error occurred in the data:\n{dt}"


class DataBasis:
    """
    A class for managing data, needed for further inheritance.
    """

    def __init__(self):
        """
        Initializes the DataBasis class.
        """
        self.data_frame = pd.DataFrame
        self.name = ""
        self.image_list = []


class Image:
    """
    A class required for the structure of X Y functions.
    """

    def __init__(self, x, y, name, distance=None):
        """
        Initializes an Function image object.

        Args:
            x: The x-value of the image.
            y: The y-value of the image.
            name: The name of the function.
            distance: The distance of the Function image (optional).

        Returns:
            None
        """
        self.x = x
        self.y = y
        self.name = name
        self.distance = distance


class TrainData(DataBasis):
    """
    A class for managing training data.
    This class inherits from the DataBasis class.
    """

    def __init__(self, filename):
        """
        Initializes a TrainData object with the data from a CSV file.

        Args:
            filename: The filename of the CSV file.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialization of the DataBasis class
            self.data_frame = pd.read_csv(filename)  # Read the CSV file into a DataFrame
            self.image_list = []  # Initialize an empty Function image list
            for dataset in self.data_frame:
                if 'x' != dataset:
                    x = self.data_frame['x'].values  # Extract the x-values
                    y = self.data_frame[dataset].values  # Extract the y-values
                    name = dataset  # Extract the name of the dataset
                    self.image_list.append(Image(x, y, name))  # Add an Function image object to the image list
        except FileNotFoundError:
            print("The specified file was not found.")
        except Exception as e:
            print("Error initializing TrainData:", str(e))

    def __iter__(self):
        """
        Initializes the iterator for the TrainData class.

        Returns:
            self: The iterator itself.
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Retrieves the next element of the iterator.

        Returns:
            value: The next element of the iterator.

        Raises:
            StopIteration: When the end of the iterator is reached.
        """
        try:
            if self.index >= len(self.image_list):
                raise StopIteration
            value = self.image_list[self.index]
            self.index += 1
            return value
        except StopIteration:
            raise
        except Exception as e:
            print("Error retrieving the next element:", str(e))



class IdealData(DataBasis):
    """
    A class for managing ideal data.
    This class inherits from the DataBasis class.
    """

    def __init__(self, filename):
        """
        Initializes an IdealData object with the data from a CSV file.

        Args:
            filename: The filename of the CSV file.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialization of the DataBasis class
            self.data_frame = pd.read_csv(filename)  # Read the CSV file into a DataFrame
            self.image_list = []  # Initialize an Function empty image list
            for dataset in self.data_frame:
                if 'x' != dataset:
                    x = self.data_frame['x'].values  # Extract the x-values
                    y = self.data_frame[dataset].values  # Extract the y-values
                    name = dataset  # Extract the name of the dataset
                    self.image_list.append(Image(x, y, name))  # Add an Image object to the Function image list
        except FileNotFoundError:
            print("The specified file was not found.")
        except Exception as e:
            print("Error initializing IdealData class:", str(e))

    def __iter__(self):
        """
        Initializes the iterator for the IdealData class.

        Returns:
            self: The iterator itself.
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Retrieves the next element of the iterator.

        Returns:
            value: The next element of the iterator.

        Raises:
            StopIteration: When the end of the iterator is reached.
        """
        try:
            if self.index >= len(self.image_list):
                raise StopIteration
            value = self.image_list[self.index]
            self.index += 1
            return value
        except StopIteration:
            raise
        except Exception as e:
            print("Error iterating over IdealData class:", str(e))


class TestData(DataBasis):
    """
    A class for managing test data.
    This class inherits from the DataBasis class.
    """

    def __init__(self, filename):
        """
        Initializes a TestData object with the data from a CSV file.

        Args:
            filename: The filename of the CSV file.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialization of the DataBasis class
            self.data_frame = pd.read_csv(filename)  # Read the CSV file into a DataFrame
            self.image_list = []  # Initialize an Function empty image list
            for dataset in self.data_frame:
                if 'x' != dataset:
                    x = self.data_frame['x'].values  # Extract the x-values
                    y = self.data_frame[dataset].values  # Extract the y-values
                    name = dataset  # Extract the name of the dataset
                    self.image_list.append(Image(x, y, name))  # Add an Function Image object to the image list
        except FileNotFoundError:
            print("The specified file was not found.")
        except Exception as e:
            print("Error initializing TestData class:", str(e))

    def __iter__(self):
        """
        Initializes the iterator for the TestData class.

        Returns:
            self: The iterator itself.
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Retrieves the next element of the iterator.

        Returns:
            value: The next element of the iterator.

        Raises:
            StopIteration: When the end of the iterator is reached.
        """
        if self.index >= len(self.image_list):
            raise StopIteration
        value = self.image_list[self.index]
        self.index += 1
        return value


class BestFits(DataBasis):
    """
    A class for managing the best fits.
    This class inherits from the DataBasis class.
    """

    def __init__(self, train_df):
        """
        Initializes a BestFits object with the training data.

        Args:
            train_df: The DataFrame of the training data.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialization of the DataBasis class
            self.data_frame_train_data = train_df  # DataFrame of the training data
            self.data_frame_bestfit = pd.DataFrame()  # DataFrame for the best fits
            self.image_list_bestfit = []  # List of the best fits (images)
            self.image_list_traindata = []  # List of the training data (images)
            for dataset in self.data_frame_train_data:
                if 'x' != dataset:
                    x = self.data_frame_train_data['x'].values  # Extract the x-values
                    y = self.data_frame_train_data[dataset].values  # Extract the y-values
                    name = dataset  # Extract the name of the dataset
                    self.image_list_traindata.append(Image(x, y, name))  # Add an Image object to the training data list
        except FileNotFoundError:
            print("The specified file was not found.")
        except Exception as e:
            print("Error initializing BestFits class:", str(e))

    def add_data_to_bestfit(self, image):
        """
        Adds a fit to the list of best fits.

        Args:
            image: The Function Image object to be added.

        Returns:
            None
        """
        self.image_list_bestfit.append(Image(image.x, image.y, image.name))

    def fill_bestfit_to_df(self):
        """
        Fills the best fit DataFrame with the found fits.

        Returns:
            None
        """
        try:
            for ideal_image in self.image_list_bestfit:
                self.data_frame_bestfit = pd.concat(
                    [self.data_frame_bestfit, pd.DataFrame({'x': ideal_image.x})]).drop_duplicates(subset='x',
                                                                                                    keep='first')
                self.data_frame_bestfit = pd.concat(
                    [self.data_frame_bestfit, pd.DataFrame({ideal_image.name: ideal_image.y})], axis=1)
        except pd.errors.EmptyDataError:
            print("The best fit DataFrame is empty.")
        except pd.errors.ConcatenateError:
            print("Error concatenating DataFrames while filling data_frame_bestfit.")
        except pd.errors.DuplicateLabelError:
            print("Duplicate labels were found when filling the BestFits DataFrame.")
        except pd.errors.MergeError:
            print("Error merging the DataFrames in the BestFits DataFrame.")
        except pd.errors.ParserError:
            print("Error parsing the data in the BestFits DataFrame.")
        except pd.errors.EmptyDataError:
            print("The BestFits DataFrame is empty.")
        except Exception as e:
            print("A general error occurred while filling the BestFits DataFrame:", str(e))




class CheckMappingFit(DataBasis):
    """
    A class for checking the mapping of the fit.
    This class inherits from the DataBasis class.
    """

    def __init__(self, test_data, best_fit):
        """
        Initializes a CheckMappingFit object with test data and the best fit.

        Args:
            test_data: The TestData object.
            best_fit: The BestFits object.

        Returns:
            None
        """
        try:
            DataBasis.__init__(self)  # Initialization of the DataBasis class
            self.data_frame_test_data = test_data.data_frame  # DataFrame of the test data
            self.data_frame_bestfit = best_fit.data_frame_bestfit  # DataFrame of the best fit
            self.data_frame_matching = pd.DataFrame()  # DataFrame for the matching matches

            self.image_list_bestfit = best_fit.image_list_bestfit  # List of the best fits (images)
            self.image_list_testdata = []  # List of the test data (images)
            self.image_list_matching_points = []  # List of the matching points (images)
            self.distance = None  # Distance
        except FileNotFoundError:
            print("The specified file was not found.")
        except Exception as e:
            print("Error initializing CheckMappingFit class:", str(e))

    def add_match_point(self, image):
        """
        Adds a matching point to the list of matching points.

        Args:
            image: The Image object to be added.

        Returns:
            None
        """
        try:
            self.image_list_matching_points.append(image)
        except Exception as e:
            print("Error adding a matching point:", str(e))

    def fill_match_to_df(self):
        """
        Fills the matching match DataFrame with the matching points.

        Returns:
            None
        """
        try:
            for ideal_image in self.image_list_matching_points:
                tt = pd.DataFrame([[ideal_image.x, ideal_image.y, ideal_image.name, ideal_image.distance]],
                                  columns=['X (Test Data)', 'Y (Test Data)', 'Matching Ideal Function', 'Deviation'])
                self.data_frame_matching = pd.concat([self.data_frame_matching, tt])
        except pd.errors.ConcatenateError:
            print("Error concatenating DataFrames while filling data_frame_matching.")
        except Exception as e:
            print("An error occurred while filling data_frame_matching:", str(e))



class QuadraticFitting:
    """
    A class for performing quadratic fitting.
    """

    def __init__(self):
        """
        Initializes a QuadraticFitting object.
        A default constructor for the contained @staticmethod.

        Returns:
            None
        """
        pass

    @staticmethod
    def calculate_best_fits(train_data, ideal_functions):
        """
        Determines the best fits for the training data and the ideal functions.

        Args:
            train_data: The TrainData object.
            ideal_functions: A list of ideal functions.

        Returns:
            best_fits: The BestFits object with the best fits.
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
                        # Error solving the linear system using numpy
                        print("Error solving the linear system:", str(np_error))

                best_fits.add_data_to_bestfit(found_fit)

            best_fits.fill_bestfit_to_df()
            return best_fits

        except pd.errors.PandasError as pd_error:
            # Error processing pandas data structures
            print("Error processing pandas data structures:", str(pd_error))

        except DataErrorException as e:
            print("An error occurred:", str(e))

        except Exception as e:
            # General error
            print("General error while executing the bestfit method:", str(e))

    @staticmethod
    def display_task_one_result(best_fit):
        """
        Displays the results for task one.

        Parameters:
            best_fit (BestFit): The BestFit instance with the best mappings.

        Returns:
            None
        """
        try:
            for i, train_data in enumerate(best_fit.image_list_traindata):
                p = figure(title=f'Best Fits for {best_fit.image_list_bestfit[i].name}', x_axis_label='x',
                           y_axis_label='y', width=1200, height=900)
                train_x = train_data.x
                train_y = train_data.y

                # Create plot for training data
                # p.scatter(train_x, train_y, legend_label='Train Data', color='blue', size=5)
                p.line(train_x, train_y, legend_label='Train Data', color='blue', line_width=4)
                x = best_fit.image_list_bestfit[i].x
                y = best_fit.image_list_bestfit[i].y

                # Add line for ideal function
                p.line(x, y, legend_label=f'Ideal Function {best_fit.image_list_bestfit[i].name}', color='red',
                       line_width=2)

                output_file(f'best_fit_{i + 1}.html')
                show(p)
                p = None
        except Exception as e:
            # General error
            print("Error displaying the plot in task one:", str(e))

    @staticmethod
    def search_y_at_x(x, ideal_data):
        """
        Searches for the y-value for a given x-value in the ideal data.

        Parameters:
            x (float): The x-value to search for.
            ideal_data (DataFrame): The ideal data.

        Returns:
            float: The corresponding y-value.
        """
        try:
            s_key = ideal_data["x"] == x
            return ideal_data.loc[s_key].iat[0, 1]
        except IndexError:
            # x-value not found
            print("The specified x-value was not found.")
        except Exception as e:
            # General error
            print("Error searching for the y-value:", str(e))

    @staticmethod
    def display_task_two_result(bestm):
        """
        Displays the results for task two.

        Parameters:
            bestm (BestM): The BestM instance with the results.

        Returns:
            None
        """
        try:
            figures = {}  # Dictionary to store the figures

            # Loop over the matching points
            for train_data in bestm.image_list_matching_points:
                figure_title = train_data.name
                html_name = f'{figure_title}.html'

                if figure_title not in figures:
                    figures[figure_title] = {
                        "figure": figure(
                            title=str("Matching Test Points for Ideal Function " + train_data.name),
                            x_axis_label='x', y_axis_label='y', width=1200, height=900),
                        "html": html_name
                    }

                figure_data = figures[figure_title]

                try:
                    # Create DataFrame for the ideal function
                    ideal_func_df = pd.DataFrame(
                        {'x': bestm.data_frame_bestfit["x"], 'y': bestm.data_frame_bestfit[figure_title]})

                    # Create scatter plot
                    scatter = figure_data["figure"].scatter(train_data.x, train_data.y, fill_color="red",
                                                            legend_label="Test point", size=8)

                    # Add hover tool to display information on hover
                    hover = HoverTool(renderers=[scatter], tooltips=[('Deviation', f'{train_data.distance}')],
                                      mode='mouse')
                    figure_data["figure"].add_tools(hover)

                    # Add line for the ideal function
                    figure_data["figure"].line(ideal_func_df["x"], ideal_func_df["y"], legend_label=figure_title,
                                               color='blue')
                except Exception as e:
                    print("Error creating the figures:", str(e))

            # Display the figures
            for figure_data in figures.values():
                try:
                    output_file(figure_data["html"])
                    show(figure_data["figure"])
                except Exception as e:
                    print("Error displaying the figures:", str(e))
        except Exception as e:
            print("Error executing 'display_task_two_result':", str(e))

    @staticmethod
    def perform_task_two(best_fit, test_data):
        """
        Performs task two by determining the best matches between test data points and ideal functions.

        Parameters:
            best_fit (BestFit): The BestFit instance with the best mappings.
            test_data (List[Point]): The list of test data points.

        Returns:
            CheckMappingFit: The CheckMappingFit instance with the best matches.
        """
        try:
            best_match = CheckMappingFit(test_data, best_fit)

            # Loop over the test data points
            for point in test_data:
                point_x = point.x
                point_y = point.y
                # Loop for individual examination of the test data points
                for x, y in zip(point_x, point_y):
                    actual_match = None
                    actual_distance = None

                    # Loop over the ideal functions and associated training data
                    for ideal_function, train_f in zip(best_fit.image_list_bestfit,
                                                       best_fit.image_list_traindata):
                        try:
                            # Create DataFrames for the training and best fits data
                            train_data_df = pd.DataFrame({'x': best_fit.data_frame_train_data["x"],
                                                          'y': best_fit.data_frame_train_data[train_f.name]})
                            best_fit_df = pd.DataFrame({'x': best_fit.data_frame_bestfit["x"],
                                                        'y': best_fit.data_frame_bestfit[ideal_function.name]})

                            # Search for the y-value at the given x position
                            locate_y = QuadraticFitting.search_y_at_x(x, best_fit_df)

                            # Calculate the deviations
                            distance = abs(locate_y - y)
                            matrix_subtraction = train_data_df - best_fit_df
                            matrix_subtraction["y"] = matrix_subtraction["y"].abs()
                            largest_divergence = math.sqrt(2) * max(matrix_subtraction["y"])

                            # Check for best match
                            if abs(distance < largest_divergence):
                                if actual_match is None or distance < actual_distance:
                                    actual_match = ideal_function.name
                                    actual_distance = distance
                        except Exception as e:
                            print("Error during calculation:", str(e))

                    if actual_match is not None:
                        best_match.add_match_point(Image(x, y, actual_match, actual_distance))

            best_match.fill_match_to_df()
            return best_match
        except DataErrorException as e:
            print("Error occurred:", str(e))
        except Exception as e:
            print("Error executing 'perform_task_two':", str(e))

    # Custom exception class
