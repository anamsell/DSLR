import MLKit

if __name__ == "__main__":
    default_target_column = "Hogwarts House"
    default_model = "train.mlmodel"

    MLKit.command_line.CommandLine.register_flag("Y", description="The target column where the values should be predicted.", default_value=default_target_column)
    MLKit.command_line.CommandLine.register_flag("m", description="The model file name used to predict values.", default_value=default_model)
    MLKit.command_line.CommandLine.register_flag("s", description="The csv file name with the predicted values.")
    MLKit.command_line.CommandLine.register_usage("log_reg_predict.py [csv_file_name]\nPredict the value of a csv data file from a trained model.")
    MLKit.command_line.CommandLine.show_usage_if_needed()

    file_name = MLKit.command_line.CommandLine.get_argument_at_index(1)
    target_column_name = MLKit.command_line.CommandLine.get_value_for_flag("Y")
    model_file_name = MLKit.command_line.CommandLine.get_value_for_flag("m")
    output_csv = MLKit.command_line.CommandLine.get_value_for_flag("s")

    data_table = MLKit.data_table.DataTable(file_name)
    model = MLKit.file_manager.get_model_data(model_file_name)
    data_table.compute_columns_attributes(model=model)
    data_table.predict(target_column_name, model)

    if output_csv is None:
        data_table.save()
    else:
        data_table.save(file_name=output_csv)
