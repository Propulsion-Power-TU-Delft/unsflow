import numpy as np


class Driver:
    """
    this class contains the driver of the Spakovszky model instability calculation.
    """
    def __init__(self, compressor_type = None):
        self.n_components = 0
        self.compressor_type = compressor_type
        self.components = []


    def add_component(self, component):
        """
        add a component to the compression system. Follow streamwise order
        """
        self.components.append(component)
        self.n_components += 1
