from enum import Enum
import MLKit


class TextAttribute:

    @staticmethod
    def start_tag(color):
        raise NotImplementedError()

    @staticmethod
    def end_tag():
        return "\033[0m"


class Color(Enum):

    red = 0
    yellow = 1
    blue = 2
    green = 3

    @staticmethod
    def start_tag(color):
        if color == Color.red:
            return "\033[91m"
        elif color == Color.yellow:
            return "\033[93m"
        elif color == Color.blue:
            return "\033[94m"
        elif color == Color.green:
            return "\033[92m"
        else:
            return ""


class Style(Enum):

    bold = 0
    underline = 0

    @staticmethod
    def start_tag(color):
        if color == Style.bold:
            return "\033[1m"
        elif color == Style.underline:
            return "\033[4m"
        else:
            return ""


def error(message, should_exit=True):
    print(attributed_str("ERROR: ", [Color.red]) + message)
    if should_exit:
        exit()


def warning(message):
    print(attributed_str("WARNING: ", [Color.yellow]) + message)


def success(message):
    print(attributed_str("Success: ", [Color.green]) + message)


def attributed_str(string, attributes):
    color = None
    style = None

    for attribute in attributes:
        if isinstance(attribute, Color):
            color = attribute
        elif isinstance(attribute, Style):
            style = attribute
    
    return Color.start_tag(color) + Style.start_tag(style) + string + TextAttribute.end_tag()


def line_str(length, character="-"):
    final_str = ""

    for _ in range(0, length):
        final_str += character
    
    return final_str


def sized_str(string, size, margin_str=" "):
    """Returns a string sized with spaces. Truncates the string in its middle if the string length is greatter than the size."""
    final_str = string

    if len(string) > size:
        first_part = final_str[:int((size - 5) / 2)]
        second_part = final_str[-int((size - 5) / 2):]
        final_str = margin_str + margin_str + first_part + "..." + second_part

    for _ in range(0, size - len(final_str)):
        final_str = margin_str + final_str
    
    return final_str
