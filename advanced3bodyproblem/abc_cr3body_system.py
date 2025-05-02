"""Abstract Base Class for 3 body problem."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy.optimize import newton


class CR3BodySystem(ABC):
    """Abstract Base Class to define a 3 body problem system."""

    def __init__(self, mass1: float, mass2: float, distance: float):
        """Initialize the CR3BodySystem class.

        Parameters
        ----------
        mass1: float
            Mass of the first primary body.
        mass2: float
            Mass of the second primary body.
        distance: float
            Distance between the two primary bodies.
        """
        self.mass1 = mass1
        self.mass2 = mass2
        self.distance = distance

        self.mass_parameter = mass2 / (mass1 + mass2)

        self.adim_pos1 = np.array([-self.mass_parameter, 0.0, 0.0])
        self.adim_pos2 = np.array([1.0 - self.mass_parameter, 0.0, 0.0])
        self.pos1 = self.adim2dim_len(self.adim_pos1)
        self.pos2 = self.adim2dim_len(self.adim_pos2)

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

    def compute_generic_point(self,
                initial_guess: Union[float, np.ndarray]) -> np.ndarray:
        """Compute the Lagrange point.

        Returns
        -------
        np.ndarray
            Lagrange point in adimensional coordinates.
        """
        sol = newton(
            self._dUdx, initial_guess, fprime=self._diff_dUdx,
            args=(self,),
        )
        if isinstance(sol, float):
            return np.array([sol, 0.0, 0.0])

        return sol

    @staticmethod
    @abstractmethod
    def _dUdx(x: Union[float, np.ndarray],  # noqa: N802
                sys: CR3BodySystem) -> Union[float, np.ndarray]:
        """Calculate dUdx - the derivative of the potential wrt x."""

    @staticmethod
    @abstractmethod
    def _diff_dUdx(x: Union[float, np.ndarray],  # noqa: N802
                    sys: CR3BodySystem) -> Union[float, np.ndarray]:
        """Calculate the derivative of dUdx."""
