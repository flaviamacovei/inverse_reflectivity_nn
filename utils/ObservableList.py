from typing import Iterable

class ObservableList(Iterable):
    """
    Observable list with explicit append and remove actions.

    Attributes:
        elements: underlying list.
        append_action: action to perform on append.
        remove_action: action to perform on remove.

    Methods:
        append: Perform append action and append element.
        remove: Perform remove action and remove element.
    """
    def __init__(self):
        """Initialise an ObservableList instance."""
        self.elements = []
        self.append_action = None
        self.remove_action = None

    def __iter__(self):
        """Return iterator over elements."""
        return iter(self.elements)

    def append(self, element):
        """Perform append action and append element."""
        if self.append_action is not None:
            self.append_action(element)
        self.elements.append(element)

    def remove(self, element):
        """Perform remove action and remove element."""
        if self.remove_action is not None:
            self.remove_action(element)
        self.elements.remove(element)