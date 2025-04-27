"""Code to determine natural and artificial Lagrange points."""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import newton


class LagrangePoints:
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

        self._set_triangular_points()
        self._set_colliner_points()

    def _set_triangular_points(self) -> None:
        """Set the triangular Lagrange points."""
        xloc = 0.5 + self.mass_parameter
        yloc = np.sqrt(3) / 2.0

        self.adim_l4 = np.array([xloc, yloc, 0.0])
        self.adim_l5 = np.array([xloc, -yloc, 0.0])

        self.l4 = self.adim2dim_len(self.adim_l4)
        self.l5 = self.adim2dim_len(self.adim_l5)

    def _set_colliner_points(self) -> None:
        """Set the collinear Lagrange points."""
        initial_guesses = np.array([-2 * self.mass_parameter,
                                    1 - 2 * self.mass_parameter, 1])
        self.adim_l1 = np.array([newton(
            self._dUdx, initial_guesses[0],
            args=(self.mass_parameter,),
        ), 0.0, 0.0])

        self.adim_l2 = np.array([newton(
            self._dUdx, initial_guesses[1],
            args=(self.mass_parameter,),
        ), 0.0, 0.0])

        self.adim_l3 = np.array([newton(
            self._dUdx, initial_guesses[2],
            args=(self.mass_parameter,),
        ), 0.0, 0.0])

        self.l1 = self.adim2dim_len(self.adim_l1)
        self.l2 = self.adim2dim_len(self.adim_l2)
        self.l3 = self.adim2dim_len(self.adim_l3)

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

    # [ ] Close but still not correct, fix this.
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
            1 - (1 - mu) / r1**3 + 3 * (x + mu) * (1 - mu) / r1**4 * np.sign(r1)
            - mu / r2**3 + 3 * (x - (1 - mu)) * mu / r2**4 * np.sign(r2)
        )
