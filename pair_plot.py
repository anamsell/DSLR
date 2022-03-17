import MLKit

if __name__ == "__main__":
    default_target_column = "Hogwarts House"
    defaut_features = ["Astronomy", "Herbology", "Ancient Runes"]

    MLKit.CommandLine.register_flag("X", description="The features to display.", default_value=defaut_features, has_multiple_values=True)
    MLKit.CommandLine.register_flag("Y", description="The target column of the values to display.", default_value=default_target_column)
    MLKit.CommandLine.register_usage("logreg_train.py [csv_file_name]\nDisplay a pair plot of a csv data set.")
    MLKit.CommandLine.show_usage_if_needed()

    file_name = MLKit.CommandLine.get_argument_at_index(1)
    target_column_name = MLKit.CommandLine.get_value_for_flag("Y")
    features = MLKit.CommandLine.get_value_for_flag("X")

    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()    
    data_table.display_pair_plot(target_column_name, features)
