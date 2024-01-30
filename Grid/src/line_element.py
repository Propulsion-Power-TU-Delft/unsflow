import numpy as np
import matplotlib.pyplot as plt


class LineElement:
    """
    Class of Line object, storing data related to a line element.
    """

    def __init__(self, z1, r1, z2, r2):
        """
        Builds the Line element object, given the coordinates of the first and secont vertex. The line is ordered therefore
        from 1 to 2.
        :param z1: z cordinate of first vertex
        :param r1: r cordinate of first vertex
        :param z2: z cordinate of second vertex
        :param r2: r cordinate of second vertex
        """
        self.z = np.array([z1, z2])
        self.r = np.array([r1, r2])

        self.z_cg = np.mean(self.z)
        self.r_cg = np.mean(self.r)

        self.l_vec = np.array([z2 - z1, r2 - r1])  # vector of the line, pointing from 1 to 2
        self.l_norm = np.linalg.norm(self.l_vec)  # line length
        self.l_tan_dir = self.l_vec / self.l_norm  # direction of the line vector, from 1 to 2
        self.l_orth_dir = np.array(
            [self.l_vec[1], -self.l_vec[0]])  # direction of the orthogonal line vector, +90deg in clockwise direction

    def plot_line_element(self):
        """
        Plot the characteristic of a line element
        """
        plt.figure()
        plt.plot(self.z, self.r, 'k', linewidth=0.5)
        plt.scatter(self.z[0], self.r[0], label='1')
        plt.scatter(self.z[1], self.r[1], label='2')
        plt.scatter(self.z_cg, self.r_cg, label='c')
        plt.quiver(self.z_cg, self.r_cg, self.l_orth_dir[0], self.l_orth_dir[1])
        plt.legend()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r$')
        plt.gca().set_aspect('equal')


