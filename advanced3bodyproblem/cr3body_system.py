"""Code to determine natural and artificial Lagrange points."""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import newton


class CR3BodySystem:
    """Class to determine natural and artificial Lagrange points."""

    def __init__(self, mass1: float, mass2: float, distance: float):
        """Initialize the LagrangePoints class.

        Parameters
        ----------
            mass1: float
                Mass of the first body.
            mass2: float
                Mass of the second body.
            distance: float
                Distance between the two bodies.
        """
        self.mass1 = mass1
        self.mass2 = mass2
        self.distance = distance

        self.mass_parameter = mass2 / (mass1 + mass2)

        self.adim_pos1 = np.array([-self.mass_parameter, 0.0, 0.0])
        self.adim_pos2 = np.array([1.0 - self.mass_parameter, 0.0, 0.0])
        self.pos1 = self.adim2dim_len(self.adim_pos1)
        self.pos2 = self.adim2dim_len(self.adim_pos2)

        self.adim_l1 = None
        self.adim_l2 = None
        self.adim_l3 = None
        self.adim_l4 = None
        self.adim_l5 = None

        self.l1 = None
        self.l2 = None
        self.l3 = None
        self.l4 = None
        self.l5 = None

    def compute_lagrange_points(self) -> None:
        """Compute the Lagrange points."""
        self.get_l1()
        self.get_l2()
        self.get_l3()
        self.get_l4()
        self.get_l5()

    def get_l1(self) -> np.ndarray:
        """Get the L1 Lagrange point.

        Returns
        -------
        np.ndarray
            L1 Lagrange point in dimensional coordinates.
        """
        if self.l1 is None:
            self.adim_l1 = self._compute_collinear_point(1 - 2 * self.mass_parameter)
            self.l1 = self.adim2dim_len(self.adim_l1)
        return self.l1

    def get_l2(self) -> np.ndarray:
        """Get the L2 Lagrange point.

        Returns
        -------
        np.ndarray
            L2 Lagrange point in dimensional coordinates.
        """
        if self.l2 is None:
            self.adim_l2 = self._compute_collinear_point(1)
            self.l2 = self.adim2dim_len(self.adim_l2)
        return self.l2

    def get_l3(self) -> np.ndarray:
        """Get the L3 Lagrange point.

        Returns
        -------
        np.ndarray
            L3 Lagrange point in dimensional coordinates.
        """
        if self.l3 is None:
            self.adim_l3 = self._compute_collinear_point(-2 * self.mass_parameter)
            self.l3 = self.adim2dim_len(self.adim_l3)
        return self.l3

    def get_l4(self) -> np.ndarray:
        """Get the L4 Lagrange point.

        Returns
        -------
        np.ndarray
            L4 Lagrange point in dimensional coordinates.
        """
        if self.l4 is None:
            self.adim_l4 = self._compute_triangular_point()
            self.l4 = self.adim2dim_len(self.adim_l4)
        return self.l4

    def get_l5(self) -> np.ndarray:
        """Get the L5 Lagrange point.

        Returns
        -------
        np.ndarray
            L5 Lagrange point in dimensional coordinates.
        """
        if self.l5 is None:
            self.adim_l5 = self._compute_triangular_point()
            self.adim_l5[1] *= -1  # to get the lower point
            self.l5 = self.adim2dim_len(self.adim_l5)
        return self.l5

    def adim2dim_len(self, adim: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert adimensional length to dimensional length.

        Parameters
        ----------
        adim: float | np.ndarray
            Adimensional length

        Returns
        -------
        float float | np.ndarray
            Dimensional length

        """
        return adim * self.distance

    def dim2adim_len(self, dim: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert dimensional length to adimensional length.

        Parameters
        ----------
        adim: float | np.ndarray
            Dimensional length

        Returns
        -------
        float float | np.ndarray
            aimensional length

        """
        return dim / self.distance

    def _compute_collinear_point(self, initial_guess: float) -> np.array:
        """Set the collinear Lagrange point.

        Returns
        -------
        np.ndarray
            Collinear Lagrange point in adimensional coordinates.
        """
        return np.array([newton(
            self._dUdx, initial_guess, fprime=self._diff_dUdx,
            args=(self.mass_parameter,),
        ), 0.0, 0.0])

    def _compute_triangular_point(self) -> np.array:
        """Set the upper (y-positive) triangular Lagrange point.

        Returns
        -------
        np.ndarray
            Collinear Lagrange point in adimensional coordinates.
        """
        xloc = 0.5 + self.mass_parameter
        yloc = np.sqrt(3) / 2.0

        return np.array([xloc, yloc, 0.0])

    @staticmethod
    def _dUdx(x: float, mu: float) -> float:  # noqa: N802
        """Calculate dUdx - the derivative of the potential wrt x.

        Funcion to be passed to the newton solver to find the collinear points.

        Parameters
        ----------
        x: float
            x coordinate
        mu: float
            mass parameter

        Returns
        -------
        float
            dUdx evaluated at x

        """
        r1 = np.abs(x - (-mu))
        r2 = np.abs(x - (1 - mu))

        return x - (1 - mu) / r1**3 * (x + mu) - mu / r2**3 * (x - (1 - mu))

    @staticmethod
    def _diff_dUdx(x: float, mu: float) -> float:  # noqa: N802
        """Calculate the derivative of dUdx.

        Funcion to be passed to the newton solver to find the collinear points.

        Parameters
        ----------
        x: float
            x coordinate
        mu: float
            mass parameter

        Returns
        -------
        float
            derivative of dUdx evaluated at x

        """
        r1 = x - (-mu)
        r2 = x - (1 - mu)

        return (
            1 - (1 - mu) / np.abs(r1)**3 + 3 * (x + mu) * (1 - mu) / r1**4 * np.sign(r1)
            - mu / np.abs(r2)**3 + 3 * (x - (1 - mu)) * mu / r2**4 * np.sign(r2)
        )
