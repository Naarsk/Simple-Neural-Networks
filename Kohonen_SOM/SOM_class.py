import warnings

import matplotlib.pyplot as plt
import numpy as np


def gauss_distance(i1, i2, s):  # gaussian weighting function
    i1 = np.array(i1)
    i2 = np.array(i2)
    return np.exp(-sum((i1 - i2) ** 2) / (2 * s ** 2))


class SOM:

    def __init__(self, dat, net_shape : tuple[int, int]):
        """
        Class that defines the network Self Organizing Map
        :param dat:
        :param net_shape:
        """
        # Initializing arrays
        self.dat = np.array(dat)
        self.M = np.shape(self.dat)[0]
        self.N = np.shape(self.dat)[1]
        self.net_shape = net_shape
        self.C = tuple(list(self.net_shape) + [self.N])
        self.w = np.array(self.C)

        # Loading default settings
        self._default_settings = {'H': 0.2, 'S': 18, 'VH': 0.993, 'VS': 0.960, 'UMAX': 10000, 'VMAX': 1000,
                                  'HMIN': 1 / 800.0}
        self.H = self._default_settings['H']
        self.S = self._default_settings['S']
        self.VH = self._default_settings['VH']
        self.VS = self._default_settings['VS']

        self.UMAX = self._default_settings['UMAX']
        self.VMAX = self._default_settings['VMAX']
        self.HMIN = self._default_settings['HMIN']

    def generate_weights(self, **kwargs):
        default_kwargs = {'seed': 0, 'low': 0.0, 'high': 1.0}
        kwargs = {**default_kwargs, **kwargs}
        np.random.seed(kwargs['seed'])
        self.w = np.random.uniform(size=self.C, low=kwargs['low'], high=kwargs['high'])

    def change_settings(self, **kwargs):

        # updating settings
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                warnings.warn("skipping non existing setting", category=UserWarning)

    def train_network(self, **kwargs):

        self.change_settings(**kwargs)

        h = self.H
        s = self.S

        for u in range(self.UMAX):

            p = self.dat[np.random.randint(self.M), :]  # random example
            self.w = self.hebbs_rule(p, h, s)  # Hebb's rule

            # EXIT CLAUSES & DATA WRITING
            if u % (self.UMAX / self.VMAX) == 0:
                h = self.VH * h
                s = self.VS * s
                if h < self.HMIN:
                    print("step size is almost nough:", h)
                    print("network did", u, "cycles")
                    break

            if u == self.UMAX - 1:
                print("no. of cycles reached", self.UMAX)

    def plot_network(self, **kwargs):
        """
        If a plotting style is available, plots the network
        :param kwargs:
        :return:
        """
        if len(self.net_shape) == 2:
            if self.N == 2:
                self.plot_network_2d()
            elif self.N == 3:
                self.plot_network_3d()
            else:
                print('No plot available')
        elif len(self.net_shape) == 1 and self.N == 2:
            self.plot_network_1d()
        else:
            print('No plot available')

    def export_weights(self, filename: str = "weight.npy") -> bool:
        """
        Saves the weights on a .npy file
        :param filename: The file name
        :type filename: str
        :return: Boolean value stating if the export has been successful
        :rtype: bool
        """

        if not filename.endswith(".npy"):
            filename = filename + ".npy"

        f = open(filename, "wb")
        np.save(f, self.w)
        f.close()

        print('Weight data saved at ' + filename)

        return True

    def export_datapoints(self, filename: str = "datapoints.npy") -> bool:
        """
        Saves the datapoints on a .npy file
        :param filename: The file name
        :type filename: str
        :return: Boolean value stating if the export has been successful
        :rtype: bool
        """

        if not filename.endswith(".npy"):
            filename = filename + ".npy"

        f = open(filename, "wb")
        np.save(f, self.dat)
        f.close()

        print('Weight data saved at ' + filename)

        return True

    ##TRAINING FUNCTIONS    

    def win(self, p) -> int:
        """
        - Decides which neuron in the winner
        - Returns the winner coordinates on the net

        :param p:
        :return:
        :rtype: int
        """
        d_min = float("inf")
        winner = 0
        for index in np.ndindex(self.net_shape):
            d = sum((p - self.w[index]) ** 2)  # 2-distance
            if d < d_min:
                d_min = d
                winner = index
        return winner

    def hebbs_rule(self, p, h, s):
        """
        Applies Hebb's rule

        :param p:
        :param h:
        :param s:
        :return:
        """
        iw = self.win(p)
        for index in np.ndindex(self.net_shape):
            self.w[index] = self.w[index] + h * gauss_distance(iw, index, s) * (p - self.w[index])
        return self.w

    ## PLOTTING FUNCTIONS

    def connect_points_2d(self):
        for (i, j) in np.ndindex(self.net_shape):
            wxp = self.w[i, j, 0]
            wyp = self.w[i, j, 1]

            if i <= self.net_shape[0] - 2:
                wxf0 = self.w[i + 1, j, 0]
                wyf0 = self.w[i + 1, j, 1]
                plt.plot([wxp, wxf0], [wyp, wyf0], 'r-')

            if j <= self.net_shape[1] - 2:
                wxf1 = self.w[i, j + 1, 0]
                wyf1 = self.w[i, j + 1, 1]
                plt.plot([wxp, wxf1], [wyp, wyf1], 'r-')

    def connect_points_3d(self, ax):
        for (i, j) in np.ndindex(self.net_shape):
            wxp = self.w[i, j, 0]
            wyp = self.w[i, j, 1]
            wzp = self.w[i, j, 2]

            if i <= self.net_shape[0] - 2:
                wxf0 = self.w[i + 1, j, 0]
                wyf0 = self.w[i + 1, j, 1]
                wzf0 = self.w[i + 1, j, 2]
                ax.plot3D([wxp, wxf0], [wyp, wyf0], [wzp, wzf0], 'r-')

            if j <= self.net_shape[1] - 2:
                wxf1 = self.w[i, j + 1, 0]
                wyf1 = self.w[i, j + 1, 1]
                wzf1 = self.w[i, j + 1, 2]
                ax.plot3D([wxp, wxf1], [wyp, wyf1], [wzp, wzf1], 'r-')

    def plot_network_1d(self):
        plt.close()
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.plot(self.dat[:, 0], self.dat[:, 1], '.')
        plt.plot(self.w[:, 0], self.w[:, 1], 'or-')
        plt.suptitle(r'Kohonen SOM', fontsize=14)
        plt.show()

    def plot_network_2d(self):
        plt.close()
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.plot(self.dat[:, 0], self.dat[:, 1], '.')
        plt.plot(self.w[:, :, 0], self.w[:, :, 1], 'or')
        self.connect_points_2d()
        plt.suptitle(r'Kohonen SOM', fontsize=14)
        plt.show()

    def plot_network_3d(self):
        # Creating figure
        plt.close()
        plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
        lw = np.reshape(self.w, (self.net_shape[0] * self.net_shape[1], 3))
        self.connect_points_3d(ax)
        ax.plot3D(self.dat[:, 0], self.dat[:, 1], self.dat[:, 2], '.')
        ax.plot3D(lw[:, 0], lw[:, 1], lw[:, 2], 'or')
        plt.suptitle(r'Kohonen SOM', fontsize=14)
        plt.show()
