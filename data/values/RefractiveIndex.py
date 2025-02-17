class RefractiveIndex(float):
    VALUE_SPACE = [0.12, 0.306, 1.38, 1.45, 1.65, 1.8, 2.0, 2.15, 2.25]
    def __init__(self, value):
        assert value in self.VALUE_SPACE, "Specified value does not exist as refractive index"
        self.value = value

    def __str__(self):
        return str(self.value)

    @staticmethod
    def ceil(x: float):
        return RefractiveIndex(min(RefractiveIndex.VALUE_SPACE, key=lambda y: y if y >= x else float('inf')))

    @staticmethod
    def floor(x: float):
        return RefractiveIndex(max(RefractiveIndex.VALUE_SPACE, key=lambda y: y if y <= x else float('-inf')))

    @staticmethod
    def round(x: float):
        return RefractiveIndex(min(RefractiveIndex.VALUE_SPACE, key=lambda y: abs(y - x)))