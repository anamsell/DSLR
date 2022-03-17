import matplotlib.pyplot as plt
import MLKit


if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
                "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures",
                "Charms", "Flying"]
    data = data_table.values_for_target_column_named("Hogwarts House", houses, features, scaled=False)
    fig, axs = plt.subplots(num="scatter plot", figsize=(15, 10))
    plt.xlabel(features[1])
    plt.ylabel(features[3])
    for house in houses:
        while None in data[house][features[1]]:
            del data[house][features[3]][data[house][features[1]].index(None)]
            del data[house][features[1]][data[house][features[1]].index(None)]
        while None in data[house][features[3]]:
            del data[house][features[1]][data[house][features[3]].index(None)]
            del data[house][features[3]][data[house][features[3]].index(None)]
        l1 = axs.scatter(data[house][features[1]], data[house][features[3]], label=house)
    fig.legend(houses, loc='upper right')
    plt.show()
