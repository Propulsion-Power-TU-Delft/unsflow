import numpy as np
import matplotlib.pyplot as plt
from .line_element import LineElement

class AreaElement:
    """
    Class of Area object, storing data related to an area element.
    """

    def __init__(self, z_cg, r_cg, z1, r1, z2, r2, z3, r3, z4, r4):
        """
        Builds the Area element object, given the coordinates of the CFD node the vertices delimiting its boundaries.
        The vertices are ordered in SW-SE-NE-NW counter-clock loop
        :param z_cg: z cordinate of primary node
        :param r_cg: r cordinate of primary node
        :param z1: z cordinate of first vertex
        :param r1: r cordinate of first vertex
        :param z2: z cordinate of second vertex
        :param r2: r cordinate of second vertex
        :param z3: z cordinate of third vertex
        :param r3: r cordinate of third vertex
        :param z4: z cordinate of fourth vertex
        :param r4: r cordinate of fourth vertex
        """
        self.z_cg = z_cg
        self.r_cg = r_cg
        self.z_v = np.array([z1, z2, z3, z4])
        self.r_v = np.array([r1, r2, r3, r4])
        self.compute_line_elements()

    def compute_line_elements(self):
        """
        Organize the line elements, connecting the vertices, in a 1D array, ordered 1-2, 2-3, 3-4, 4-0
        """
        self.line_elements = np.empty((4), dtype=LineElement)
        for i in range(len(self.line_elements)-1):
            self.line_elements[i] = LineElement(self.z_v[i], self.r_v[i], self.z_v[i+1], self.r_v[i+1])
        self.line_elements[-1] = LineElement(self.z_v[-1], self.r_v[-1], self.z_v[0], self.r_v[0])

    def plot_area_element(self):
        """
        Just plot the element, checking the ordering
        """
        plt.figure()
        plt.scatter(self.z_cg, self.r_cg)
        for i in range(len(self.z_v)):
            plt.scatter(self.z_v[i], self.r_v[i], label='%i' %(i+1))
            plt.plot(self.z_v[i:i+2], self.r_v[i:i+2], 'k', linewidth=0.5)
            if i==len(self.z_v)-1:
                plt.plot([self.z_v[-1], self.z_v[0]], [self.r_v[-1], self.r_v[0]], 'k', linewidth=0.5)


        plt.legend()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')



