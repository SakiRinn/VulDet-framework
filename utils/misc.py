import inspect
import warnings


def get_classes(module):
    members = inspect.getmembers(module)
    classes = {member[0]: member[1] for member in members if inspect.isclass(member[1])}
    return classes

def get_param_names(func):
    signature = inspect.signature(func)
    params = signature.parameters
    return [p for p in params if p not in ['args', 'kwargs']]


class WarningCounter:

    def __init__(self, match_text, custom_message):
        self.count = 0
        self.match_text = match_text
        self.custom_message = custom_message

    def __enter__(self):
        self._original_showwarning = warnings.showwarning
        warnings.showwarning = self._showwarning
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.showwarning = self._original_showwarning
        return False

    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        if self.match_text in str(message):
            self.count += 1
            warnings.warn(f'({self.count}) {self.custom_message}')
        else:
            self._original_showwarning(message, category, filename, lineno, file, line)
