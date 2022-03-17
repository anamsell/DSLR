import MLKit
import math

if __name__ == "__main__":
    default_features = ['Astronomy', "Herbology", 'Ancient Runes']
    default_target_column = "Hogwarts House"

    MLKit.CommandLine.register_flag("X", description="The features to train.", default_value=default_features, has_multiple_values=True)
    MLKit.CommandLine.register_flag("Y", description="The column to train.", default_value=default_target_column)
    MLKit.CommandLine.register_flag("o", description="The output file name.", default_value="train")
    MLKit.CommandLine.register_flag("l", description="The learning rate.", default_value=0.001)
    MLKit.CommandLine.register_flag("a", description="The accuracy split to train the data.")
    MLKit.CommandLine.register_usage("Build a model from a csv file.\nlogreg_train.py [train_model]")
    MLKit.CommandLine.show_usage_if_needed()

    input_file_name = MLKit.CommandLine.get_argument_at_index(1)
    output_file_name = MLKit.CommandLine.get_value_for_flag("o")
    target_column = MLKit.CommandLine.get_value_for_flag("Y")
    features = MLKit.CommandLine.get_value_for_flag("X")
    learning_rate = float(MLKit.CommandLine.get_value_for_flag("l"))
    accuracy_split = MLKit.CommandLine.get_value_for_flag("a")
    accuracy_split = None if accuracy_split is None else float(accuracy_split)

    data_table = MLKit.DataTable(input_file_name)
    data_table.compute_columns_attributes()
    data_table.train(target_column, default_features, output_file_name, learning_rate=learning_rate, accuracy_split=accuracy_split)

    if accuracy_split is not None:
        data_table.accuracy(output_file_name)
