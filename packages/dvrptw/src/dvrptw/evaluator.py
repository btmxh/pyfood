"""Objective evaluators for DVRPTW.

An evaluator collapses the two raw objectives into a single scalar that an
ILP (or any other solver) minimises:

    obj1 = total travel cost          (minimise → lower is better)
    obj2 = number of rejected requests (minimise → lower is better)

Both objectives point in the same direction: the scalar is always expressed
as a sum of non-negative penalty terms:

    scalar = f(obj1, obj2) ≥ 0

Hierarchy
---------
``Evaluator``           – Protocol: scalar(obj1, obj2) -> float
``WeightedSumEvaluator``      – w1·obj1 + w2·obj2         (raw units, no normalisation)
``StarNormEvaluator``   – normalises obj1 by Σ 2·dist(depot,c), obj2 by n_customers
``LinearNormEvaluator`` – caller supplies explicit bounds [lo1,hi1] and [lo2,hi2]

All normalising evaluators take *relative* weights (w1, w2) — they do not
need to sum to 1 but are normalised internally so the result is scale-
invariant with respect to the weight magnitudes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Evaluator(Protocol):
    """Collapse two raw objectives into a single non-negative minimisation scalar."""

    def scalar(self, obj1: float, obj2: float) -> float:
        """Return a scalar to minimise.

        Parameters
        ----------
        obj1:
            Total travel cost (non-negative).
        obj2:
            Number of rejected requests (non-negative integer).

        Returns
        -------
        float
            Scalar to *minimise*.  Lower is better.  Always non-negative.
        """
        ...

    def ilp_coefficients(
        self, star_cost: float, n_customers: float
    ) -> tuple[float, float]:
        """Return ``(coeff1, coeff2)`` for use in a PuLP minimisation objective.

        Because the ILP decision variable tracks *served* requests (not rejected),
        the objective is expressed as::

            coeff1 · travel_cost_expr  +  coeff2 · served_count_expr

        Minimising rejections = maximising service, so ``coeff2 < 0`` (rewarding
        more served customers reduces the penalty).  ``coeff1 > 0`` penalises travel.

        Parameters
        ----------
        star_cost:
            Instance-derived upper bound on travel cost: Σ_c 2·dist(depot, c).
        n_customers:
            Total number of customer requests (excluding depot).
        """
        ...


# ---------------------------------------------------------------------------
# Raw weighted sum — no normalisation
# ---------------------------------------------------------------------------


class WeightedSumEvaluator:
    """Linear combination of raw objectives with no normalisation.

        scalar = w1 · obj1 + w2 · obj2

    Both terms are non-negative penalties.  Because the two objectives live
    in different units (distance vs. count), the weight ratio ``w1/w2`` must
    be chosen to match those units.

    Parameters
    ----------
    w1 : float
        Weight on travel cost (obj1).  Must be ≥ 0.
    w2 : float
        Weight on rejected count (obj2).  Must be ≥ 0.
    """

    def __init__(self, w1: float, w2: float) -> None:
        if w1 < 0 or w2 < 0:
            raise ValueError(f"Weights must be non-negative, got w1={w1}, w2={w2}")
        self.w1 = w1
        self.w2 = w2

    def scalar(self, obj1: float, obj2: float) -> float:
        return self.w1 * obj1 + self.w2 * obj2

    def ilp_coefficients(
        self, star_cost: float, n_customers: float
    ) -> tuple[float, float]:
        """Return (coeff1, coeff2) s.t. ILP objective = coeff1·travel + coeff2·served.

        rejected = n_customers - served, so:
            w2 · rejected = w2 · n_customers - w2 · served
        The constant w2·n_customers does not affect the optimum, so the
        coefficient on the served variable is -w2.
        """
        return (self.w1, -self.w2)

    def __repr__(self) -> str:
        return f"WeightedSumEvaluator(w1={self.w1}, w2={self.w2})"


# ---------------------------------------------------------------------------
# Normalising evaluators
# ---------------------------------------------------------------------------


class LinearNormEvaluator:
    """Normalise each objective by caller-supplied bounds, then combine.

        obj1_norm = (obj1 - lo1) / (hi1 - lo1)   ∈ [0, 1]
        obj2_norm = (obj2 - lo2) / (hi2 - lo2)   ∈ [0, 1]
        scalar    = w1_rel · obj1_norm + w2_rel · obj2_norm

    where ``w1_rel`` and ``w2_rel`` are the relative weights normalised so
    that ``w1_rel + w2_rel = 1``.

    Parameters
    ----------
    w1 : float
        Relative weight on cost.  Only the ratio ``w1/(w1+w2)`` matters.
    w2 : float
        Relative weight on rejections.
    lo1, hi1 : float
        Lower and upper bounds for obj1 (travel cost).
    lo2, hi2 : float
        Lower and upper bounds for obj2 (rejected count).
    """

    def __init__(
        self,
        w1: float,
        w2: float,
        lo1: float,
        hi1: float,
        lo2: float,
        hi2: float,
    ) -> None:
        if w1 < 0 or w2 < 0:
            raise ValueError(f"Weights must be non-negative, got w1={w1}, w2={w2}")
        total = w1 + w2
        if total == 0:
            raise ValueError("At least one weight must be positive")
        self.w1 = w1 / total
        self.w2 = w2 / total
        self._lo1, self._hi1 = lo1, hi1
        self._lo2, self._hi2 = lo2, hi2
        self._range1 = hi1 - lo1 or 1.0
        self._range2 = hi2 - lo2 or 1.0

    def scalar(self, obj1: float, obj2: float) -> float:
        n1 = (obj1 - self._lo1) / self._range1
        n2 = (obj2 - self._lo2) / self._range2
        return self.w1 * n1 + self.w2 * n2

    def ilp_coefficients(
        self, star_cost: float, n_customers: float
    ) -> tuple[float, float]:
        """Return (coeff1, coeff2) for use in a PuLP minimisation objective."""
        # rejected_norm = (n - served - lo2) / range2
        # w2 · rejected_norm contributes -w2/range2 to the served coefficient
        return (self.w1 / self._range1, -self.w2 / self._range2)

    def __repr__(self) -> str:
        return (
            f"LinearNormEvaluator(w1={self.w1:.3f}, w2={self.w2:.3f}, "
            f"lo1={self._lo1}, hi1={self._hi1}, lo2={self._lo2}, hi2={self._hi2})"
        )


class StarNormEvaluator:
    """Normalise using instance-derived bounds, then combine.

    obj1 upper bound — *star-route cost*: the cost if every customer is served
    by its own dedicated round-trip from the depot.  This is the tightest
    feasible upper bound: ``C_max = Σ_c 2·dist(depot, c)``.

    obj2 upper bound — ``n_customers`` (all requests rejected).
    Both lower bounds are 0.

        obj1_norm = obj1 / C_max          ∈ [0, 1]
        obj2_norm = obj2 / n_customers    ∈ [0, 1]
        scalar    = w1_rel · obj1_norm + w2_rel · obj2_norm

    Parameters
    ----------
    w1 : float
        Relative weight on cost.
    w2 : float
        Relative weight on rejections.
    star_cost : float
        Precomputed Σ_c 2·dist(depot, c).
    n_customers : int
        Total number of customer requests (excluding depot).
    """

    def __init__(
        self,
        w1: float,
        w2: float,
        star_cost: float,
        n_customers: int,
    ) -> None:
        if w1 < 0 or w2 < 0:
            raise ValueError(f"Weights must be non-negative, got w1={w1}, w2={w2}")
        total = w1 + w2
        if total == 0:
            raise ValueError("At least one weight must be positive")
        self.w1 = w1 / total
        self.w2 = w2 / total
        self._star_cost = star_cost or 1.0
        self._n = float(n_customers) or 1.0

    @staticmethod
    def from_instance(
        w1: float,
        w2: float,
        instance: "DVRPTWInstance",  # type: ignore[name-defined]  # noqa: F821
    ) -> "StarNormEvaluator":
        """Construct directly from a ``DVRPTWInstance``, computing bounds automatically."""
        depot = next(r for r in instance.requests if r.is_depot)
        customers = [r for r in instance.requests if not r.is_depot]
        star_cost = sum(2.0 * depot.distance_to(c) for c in customers)
        return StarNormEvaluator(w1, w2, star_cost, len(customers))

    def scalar(self, obj1: float, obj2: float) -> float:
        return self.w1 * (obj1 / self._star_cost) + self.w2 * (obj2 / self._n)

    def ilp_coefficients(
        self, star_cost: float, n_customers: float
    ) -> tuple[float, float]:
        """Return (coeff1, coeff2) for use in a PuLP minimisation objective.

        Uses the evaluator's own precomputed normalisation factors.
        rejected = n - served, so the served coefficient is -w2/n.
        """
        return (self.w1 / self._star_cost, -self.w2 / self._n)

    def __repr__(self) -> str:
        return (
            f"StarNormEvaluator(w1={self.w1:.3f}, w2={self.w2:.3f}, "
            f"star_cost={self._star_cost:.2f}, n_customers={self._n:.0f})"
        )
