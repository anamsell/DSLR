import sys
import MLKit

class CommandLine:

    class Flag:

        def __init__(self, name, description, default_value, required, require_parameters, has_multiple_values):
            self.name = name
            self.description = description
            self.default_value = default_value
            self.is_required = required
            self.require_parameters = require_parameters
            self.has_multiple_values = has_multiple_values

    __usage = ""
    __flags = {}

    @staticmethod
    def show_usage_if_needed():
        """Show the registered usage of the executable if it has no arguments."""
        if len(sys.argv) > 1:
            return
        
        if len(CommandLine.__usage) > 0:
            print(CommandLine.__usage)
        if len(CommandLine.__flags.items()) > 0:
            for flag in CommandLine.__flags.values():
                print("    -" + flag.name + ": " + flag.description)
        
        exit()
    
    @staticmethod
    def register_usage(usage):
        """Register a usage for the executable."""
        CommandLine.__usage = usage

    @staticmethod
    def get_file_name():
        if len(sys.argv) < 2:
            MLKit.Display.error("Should take one argument.")
        
        return sys.argv[1]
    
    @staticmethod
    def get_argument_at_index(index):
        if len(sys.argv) < index:
            return None
        return sys.argv[index]

    @staticmethod
    def register_flag(name, description="", default_value=None, required=False, require_parameters=True, has_multiple_values=False):
        """Register a flag for the command line."""
        flag = CommandLine.Flag(name, description, default_value, required, require_parameters, has_multiple_values)
        CommandLine.__flags[name] = flag
    
    @staticmethod
    def get_value_for_flag(flag_name):
        """Returns the values for a flag in the executable arguments."""
        if not CommandLine.is_flag_registered(flag_name):
            MLKit.Display.error("Flag -" + flag_name + " doesn't exist.")

        flag = CommandLine.__flags.get(flag_name)
        args = sys.argv
        current_flag = None
        values = []

        if flag is None:
            MLKit.Display.error("Flag -" + flag_name + " is not registered.")

        for arg in args:
            if current_flag is None:
                if not CommandLine.__is_flag(arg):
                    continue
                if flag_name in arg:
                    current_flag = CommandLine.__flags[arg[1:]]
            else:
                if CommandLine.__is_flag(arg):
                    break
                values.append(arg)
        
        if current_flag is None:
            if CommandLine.__flags[flag_name].is_required:
                MLKit.Display.error("Flag -" + flag_name + " is required.")
            return CommandLine.__flags[flag_name].default_value
        
        if current_flag.require_parameters and len(values) == 0:
            MLKit.Display.error("Flag -" + flag_name + " requires parameters.")
        
        return values if current_flag.has_multiple_values else values[0]
                
    @staticmethod
    def __is_flag(string):
        if not string.startswith("-"):
            return False
        
        string = string[1:]

        return string.isalpha()

    @staticmethod
    def is_flag_registered(flag_name):
        return flag_name in CommandLine.__flags.keys()
