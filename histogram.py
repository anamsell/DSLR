import matplotlib.pyplot as plt
import MLKit


if __name__ == "__main__":
    default_target_column = "Hogwarts House"
    default_rows = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    default_features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
                "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms"
                , "Flying"]

    MLKit.CommandLine.register_flag("r", description="The row of the target column.", default_value=default_rows, has_multiple_values=True)
    MLKit.CommandLine.register_flag("X", description="The features of the histogram.", default_value=default_features, has_multiple_values=True)
    MLKit.CommandLine.register_flag("Y", description="The target column.", default_value=default_target_column)
    MLKit.CommandLine.register_usage("Display an histogram from a csv data file.")
    MLKit.CommandLine.show_usage_if_needed()

    file_name = MLKit.CommandLine.get_argument_at_index(1)
    rows = MLKit.CommandLine.get_value_for_flag("r")
    features = MLKit.CommandLine.get_value_for_flag("X")
    target_column = MLKit.CommandLine.get_value_for_flag("Y")
    
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()
    data_table.display_histogram(target_column, rows, features)
