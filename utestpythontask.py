import unittest
from dataprocessing import QuadraticFitting, TrainData, BestFits
import pandas as pd
import numpy as np


class QuadraticFittingTestCase(unittest.TestCase):
    """
    Unit tests for the QuadraticFitting class.
    """

    def setUp(self):
        # Initialize the test data and objects
        self.train_data = pd.read_csv('train.csv')
        self.ideal_functions = pd.read_csv('ideal.csv')
        self.best_fit = pd.read_csv('test.csv')

    def tearDown(self):
        # Clean up after each test case
        pass

    def test_search_y_at_x(self):
        # Tests the search_y_at_x method of the QuadraticFitting class.

        # Prepare the test data
        ideal_data = pd.DataFrame({'x': self.ideal_functions['x'], 'y': self.ideal_functions['y1']})
        x = -19.1
        expected_y = -0.2478342

        # Call the method to be tested
        y = QuadraticFitting.search_y_at_x(x, ideal_data)

        # Check the expected result
        self.assertEqual(y, expected_y)

    def test_create_traindata_structure(self):
        # Tests the create_traindata_structure method of the QuadraticFitting class.

        # Prepare the test data
        traindata = self.train_data
        test_trainData = TrainData('train.csv')

        # Check if the created data structure is correct
        self.assertTrue(test_trainData.data_frame.equals(traindata))

    def test_best_fits(self):
        # Tests the best_fits method of the QuadraticFitting class.

        # Prepare the test data and objects
        test_trainData = TrainData('train.csv')
        test_IdealData = TrainData('ideal.csv')
        test_best_fits = BestFits(test_trainData.data_frame)

        for train_data in test_trainData:
            fits = []

            # Extract the x and y values from the training data
            train_x = train_data.x
            train_y = train_data.y

            for ideal_function in test_IdealData:
                ideal_x = ideal_function.x
                ideal_y = ideal_function.y

                try:
                    # Calculate the best fit using numpy
                    m, c = np.linalg.lstsq(np.vstack([ideal_x, np.ones(len(ideal_x))]).T, ideal_y, rcond=None)[0]
                    sum_squared_diff = np.sum(np.square(train_y - (m * train_x + c)))
                    fits.append((ideal_function, sum_squared_diff))
                except np.linalg.LinAlgError as np_error:
                    # Error in solving the linear equation system using numpy
                    print("Error solving the linear equation system:", str(np_error))

            # Find the best fit
            found_fit = min(fits, key=lambda fit: fit[1])[0] if fits else None
            test_best_fits.add_data_to_bestfit(found_fit)

        # Fill the best fit into a DataFrame
        test_best_fits.fill_bestfit_to_df()

        # Execute the method to be tested
        best_fit = QuadraticFitting.calculate_best_fits(test_trainData, test_IdealData)

        # Check if the generated best fit is correct
        self.assertTrue(best_fit.data_frame_bestfit.equals(test_best_fits.data_frame_bestfit))


if __name__ == '__main__':
    # Run the unit tests
    unittest.main()
