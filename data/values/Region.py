class Region():
    """
    Region class to model an interval of wavelengths with a specific value.

    Attributes:
        start_wl: Start wavelength of the region.
        end_wl: End wavelength of the region.
        value: Value of the region.

    Methods:
        get_start_wl: Return start wavelength of the region.
        get_end_wl: Return end wavelength of the region.
        get_value: Return value of the region.
    """
    def __init__(self, start_wl, end_wl, value):
        """
        Initialise a Region instance.

        Args:
            start_wl: Start wavelength of the region.
            end_wl: End wavelength of the region.
            value: Value of the region.
        """
        assert start_wl < end_wl, "Start wavelength must be less than end wavelength"
        assert value >= 0.0 and value <= 1.0, "Value must be between 0 and 1"
        self.start_wl = start_wl
        self.end_wl = end_wl
        self.value = value

    def __str__(self):
        """Return string representation of Region object."""
        return f"Region: {int(self.start_wl * 1e3):5} nm to {int(self.end_wl * 1e3):5} nm: {self.value:2.2f}"

    def get_start_wl(self):
        """Return start wavelength of the region."""
        return self.start_wl

    def get_end_wl(self):
        """Return end wavelength of the region."""
        return self.end_wl

    def get_value(self):
        """Return value of the region."""
        return self.value