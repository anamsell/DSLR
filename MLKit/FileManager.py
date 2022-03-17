import MLKit
import csv
import json


def fd(file_name, option):
    try:
        return open(file_name, option)
    except IOError:
        MLKit.Display.error("No such file named " + file_name)


def get_content_of_file(file_name):
    file_descriptor = fd(file_name, "r")
    content = file_descriptor.readlines()
    file_descriptor.close()
    return content


def get_csv_data(file_content, delimiter=","):
    if file_content == None:
        return None
    
    columns = dict()
    csv_content = csv.DictReader(file_content, delimiter=delimiter)
    
    for row in csv_content:
        for (key, value) in row.items():
            if columns.get(key) is None:
                columns[key] = []
            try:
                if len(value) == 0:
                    columns[key].append(None)
                else:
                    columns[key].append(value)
            except TypeError:
                MLKit.Display.error("The file is not correctly formated.")

    return columns


def save_string(string, file_name):
    file_descriptor = fd(file_name, "w+")
    file_descriptor.write(string)
    file_descriptor.close()


def get_model_data(file_name):
    file_descriptor = fd(file_name, "r")
    return json.load(file_descriptor)


def save_model_data(data, file_name):
    file_descriptor = fd(file_name + ".mlmodel", "w+")
    json.dump(data, file_descriptor)
    file_descriptor.close()
