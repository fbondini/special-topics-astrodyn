"""Artificial 3 bdoy system class."""
from __future__ import annotations

from typing import Union

import numpy as np

from .abc_cr3body_system import CR3BodySystem


class ArtificialCR3BodySystem(CR3BodySystem):
    """Class to define artificial CR3 body system."""

    def __init__(self, mass1: float, mass2: float, distance: float,
                    solar_radiation_ratio: float, normal: np.ndarray = None):
        """Initialize the ArtificialCR3BodySystem class.

        Reference paper for the parameters: https://www.jstage.jst.go.jp/article/tjsass/51/174/51_174_220/_pdf

        Parameters
        ----------
        mass1: float
            Mass of the first primary body.
        mass2: float
            Mass of the second primary body.
        distance: float
            Distance between the two primary bodies.
        solar_radiation_ratio: float
            ratio of solar radiation pressure force to solar gravitational force exerted
            on sail when normal to sunlight direction.
        normal: np.ndarray
            [ ] first implementation of the code uses n=[-1, 0, 0] (sail normal to sun)
        """
        super().__init__(mass1, mass2, distance)
        self.solar_radiation_ratio = solar_radiation_ratio
        self.normal = normal

    @staticmethod
    def _dUdx(x: Union[float, np.ndarray],  # noqa: N802
                sys: ArtificialCR3BodySystem) -> Union[float, np.ndarray]:
        """Calculate dUdx - the derivative of the potential wrt x.

        Funcion to be passed to the newton solver to find the collinear points.

        Parameters
        ----------
        x: float
            x coordinate
        mu: float
            mass parameter
        sys: ArtificialCR3BodySystem
            ArtificialCR3BodySystem object

        Returns
        -------
        float
            dUdx evaluated at x

        """
        if sys.normal is None:
            return sys._artificial_collinear_dUdx(x, sys)

        return sys._artificial_generic_dUdx(x, sys)

    @staticmethod
    def _artificial_collinear_dUdx(x: float, sys: ArtificialCR3BodySystem) -> float:  # noqa: N802
        """Calculate dUdx - the derivative of the potential wrt x.

        Parameters
        ----------
        x: float
            x coordinate
        sys: ArtificialCR3BodySystem
            ArtificialCR3BodySystem object

        Returns
        -------
        float
            dUdx evaluated at x

        """
        mu = sys.mass_parameter
        beta = sys.solar_radiation_ratio

        r1 = np.abs(x - (-mu))
        r2 = np.abs(x - (1 - mu))

        return (
            x - (1 - mu) / r1**3 * (x + mu) - mu / r2**3 * (x - (1 - mu)) -
            beta * (1 - mu) / r1**2
        )

    @staticmethod
    def _artificial_generic_dUdx(x: float, sys: ArtificialCR3BodySystem) -> float:  # noqa: N802
        """Calculate dUdx - the derivative of the potential wrt x.

        Parameters
        ----------
        x: float
            x coordinate
        sys: ArtificialCR3BodySystem
            ArtificialCR3BodySystem object

        Returns
        -------
        float
            dUdx evaluated at x

        """
        mu = sys.mass_parameter
        beta = sys.solar_radiation_ratio

        r1 = np.abs(x - (-mu))
        r2 = np.abs(x - (1 - mu))

        return (
            x - (1 - mu) / r1**3 * (x + mu) - mu / r2**3 * (x - (1 - mu)) +
            beta * (1 - mu) / r1**2
        )

    @staticmethod
    def _diff_dUdx(x: Union[float, np.ndarray],  # noqa: N802
                    sys: ArtificialCR3BodySystem) -> Union[float, np.ndarray]:
        """Calculate the derivative of dUdx.

        Funcion to be passed to the newton solver to find the collinear points.

        Parameters
        ----------
        x: float
            x coordinate
        sys: ArtificialCR3BodySystem
            ArtificialCR3BodySystem object

        Returns
        -------
        float
            derivative of dUdx evaluated at x

        """
        if sys.normal is None:
            return sys._artificial_collinear_diff_dUdx(x, sys)

        msg = (
            "This method is not implemented yet for the given normal vector."
        )
        raise NotImplementedError(msg)

    @staticmethod
    def _artificial_collinear_diff_dUdx(x: float,  # noqa: N802
                                        sys: ArtificialCR3BodySystem) -> float:
        """Calculate the derivative of dUdx.

        Parameters
        ----------
        x: float
            x coordinate
        sys: ArtificialCR3BodySystem
            ArtificialCR3BodySystem object

        Returns
        -------
        float
            derivative of dUdx evaluated at x

        """
        mu = sys.mass_parameter
        beta = sys.solar_radiation_ratio

        r1 = x - (-mu)
        r2 = x - (1 - mu)

        return (
            1 - (1 - mu) / np.abs(r1)**3 + 3 * (x + mu) * (1 - mu) / r1**4 * np.sign(r1)
            - mu / np.abs(r2)**3 + 3 * (x - (1 - mu)) * mu / r2**4 * np.sign(r2)
            + 2 * beta * (1 - mu) / np.abs(r1)**3 * np.sign(r1)
        )
