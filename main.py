from dataprocessing import TrainData, TestData, IdealData, QuadraticFitting
from sqlhelper import SQLHelperClass

if __name__ == '__main__':
    # Initialize the TrainData object with 'train.csv'
    train_data = TrainData('train.csv')
    # Initialize the IdealData object with 'ideal.csv'
    ideal_data = IdealData('ideal.csv')
    # Initialize the SQLHelperClass object
    sql_helper = SQLHelperClass()

    # Clear all tables in the database
    sql_helper.clear_tables()
    # Write the training data into the database
    sql_helper.write_dataframe_into_table(train_data.data_frame, "training", " Training Data")
    # Write the ideal data into the database
    sql_helper.write_dataframe_into_table(ideal_data.data_frame, "ideal", " Ideal Data")
    # Perform the QuadraticFitting calculation
    fits_result = QuadraticFitting.calculate_best_fits(train_data, ideal_data)
    # Display the result of Task 1
    QuadraticFitting.display_task_one_result(fits_result)

    # Initialize the TestData object with 'test.csv'
    test_data = TestData('test.csv')
    # Perform Task 2 using the result from Task 1 and the test data
    best_matches = QuadraticFitting.perform_task_two(fits_result, test_data)
    # Display the result of Task 2
    QuadraticFitting.display_task_two_result(best_matches)
    # Write the match result into the database with the corresponding ideal function and the deviation
    sql_helper.write_dataframe_into_table(best_matches.data_frame_matching, "matches", " Result")
    # Output all table contents from the database using print()
    sql_helper.print_all_tables()