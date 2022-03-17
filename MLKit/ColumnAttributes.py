from enum import Enum
import math
import MLKit


class ColumnAttributes:

    class Type(Enum):

        string = 0
        numeric = 1

        @staticmethod
        def type_of_value(value):
            try:
                float(value)
                return ColumnAttributes.Type.numeric
            except ValueError:
                pass

            return ColumnAttributes.Type.string
        
        @staticmethod
        def name_of_type(column_type):
            if column_type == ColumnAttributes.Type.string:
                return "String"
            elif column_type == ColumnAttributes.Type.numeric:
                return "Numeric"
            return None
            

    def __init__(self, column):
        self.count = 0
        self.mean = None
        self.minimum = float("inf")
        self.maximum = -float("inf")
        self.percent_25 = None
        self.percent_50 = None
        self.percent_75 = None
        self.std = None
        self.type = None
        self.numeric_values = {}
        self.__compute_attributes(column)
    
    def __compute_attributes(self, column):
        values_sum = 0

        for value in column.values:
            if value == None:
                continue

            if not self.__is_type_correct_for_value(value):
                MLKit.Display.warning("Column " + column.name + " contains different value types.")
                self.minimum = None
                self.maximum = None
                return
            
            self.__transform_value_to_numeric_if_needed(value)

            float_value = self.numeric_value_for_value(value)
            values_sum += float_value
            self.count += 1

            if float_value < self.minimum:
                self.minimum = float_value
            if float_value > self.maximum:
                self.maximum = float_value
        
        if self.count == 0:
            return

        self.mean = values_sum / self.count
        self.percent_25 = 0
        self.percent_50 = 0
        self.percent_75 = 0
        squared_sum = 0

        replaced_values = []
        for value in column.values:
            if value is None:
                continue
            replaced_values.append(self.numeric_value_for_value(value))
        sorted_values = sorted(replaced_values)

        for (index, value) in enumerate(sorted_values):
            if value == None:
                continue

            squared_sum += (self.mean - value) ** 2
            
            if index == round(self.count / 4):
                self.percent_25 = value
            if index == round(self.count / 2):
                self.percent_50 = value
            if index == round(3 * self.count / 4):
                self.percent_75 = value

        variance = squared_sum / self.count
        self.std = math.sqrt(variance)
    
    def __is_type_correct_for_value(self, value):
        if value == None:
            return True
        
        if self.type == ColumnAttributes.Type.string and value == "Nan":
            return True
        elif self.type == None:
            self.type = ColumnAttributes.Type.type_of_value(value)
            return True
        elif self.type == ColumnAttributes.Type.type_of_value(value):
            return True
        else:
            return False
        
    def __transform_value_to_numeric_if_needed(self, value):
        if value == None or self.type == ColumnAttributes.Type.numeric:
            return
        
        if self.numeric_values.get(value) == None:
            self.numeric_values[value] = len(self.numeric_values.keys())
    
    def numeric_value_for_value(self, value):
        if self.type == ColumnAttributes.Type.numeric:
            return float(value)
        
        return self.numeric_values[value]

    def value_for_key(self, name):
        if name == "Type":
            return ColumnAttributes.Type.name_of_type(self.type)
        elif name == "Count":
            return self.count
        elif name == "Mean":
            return self.mean
        elif name == "Std":
            return self.std
        elif name == "Min":
            return self.minimum
        elif name == "25%":
            return self.percent_25
        elif name == "50%":
            return self.percent_50
        elif name == "75%":
            return self.percent_75
        elif name == "Max":
            return self.maximum
        else:
            return None

    @staticmethod
    def all():
        return ["Type", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
