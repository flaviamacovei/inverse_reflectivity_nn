from typing import Iterable

class ObservableList(Iterable):
    def __init__(self):
        self.elements = []
        self.append_action = None
        self.remove_action = None

    def __iter__(self):
        return iter(self.elements)

    def append(self, element):
        if self.append_action is not None:
            self.append_action(element)
        self.elements.append(element)

    def remove(self, element):
        if self.remove_action is not None:
            self.remove_action(element)
        self.elements.remove(element)