from typing import Protocol


class Plot(Protocol):


    def plot(self):...


class PseudoSpectrum:


    def plot(self):...


class RootCircle:
    """
    Plots the unity root circle of a root music engine.
    """


    def plot(self):...