class Region():
    def __init__(self, start_wl, end_wl, value):
        assert start_wl < end_wl, "Start wavelength must be less than end wavelength"
        assert value >= 0.0 and value <= 1.0, "Value must be between 0 and 1"
        self.start_wl = start_wl
        self.end_wl = end_wl
        self.value = value

    def __str__(self):
        return f"Region: {int(self.start_wl * 1e3):5} nm to {int(self.end_wl * 1e3):5} nm: {self.value:2.2f}"

    def get_start_wl(self):
        return self.start_wl

    def get_end_wl(self):
        return self.end_wl

    def get_value(self):
        return self.value