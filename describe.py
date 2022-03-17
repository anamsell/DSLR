import MLKit

if __name__ == "__main__":
    MLKit.CommandLine.register_flag("s", description="Start index of column to display.", default_value=0)
    MLKit.CommandLine.register_flag("e", description="End index of column to display.", default_value=-1)
    MLKit.CommandLine.register_usage("Display the attributes of a csv data file.")
    MLKit.CommandLine.show_usage_if_needed()

    file_name = MLKit.CommandLine.get_argument_at_index(1)
    start_index = int(MLKit.CommandLine.get_value_for_flag("s"))
    end_index = int(MLKit.CommandLine.get_value_for_flag("e"))

    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()
    data_table.display_attributes(from_index=start_index, to_index=end_index)
