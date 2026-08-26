from __future__ import annotations

import re
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


SUPPORTED_LABELS = ("unhomogenous", "polinomial", "separable")


class EquationNormalizer(BaseEstimator, TransformerMixin):
    """Normalize equation strings while preserving mathematical structure."""

    _spaces = re.compile(r"\s+")
    _latex_commands = re.compile(r"\\([a-zA-Z]+)")

    def fit(self, x: Iterable[object], y: Iterable[object] | None = None) -> "EquationNormalizer":
        return self

    def transform(self, x: Iterable[object]) -> list[str]:
        return [self._normalize(value) for value in x]

    def _normalize(self, value: object) -> str:
        equation = "" if value is None else str(value)
        equation = equation.lower()
        equation = equation.replace("−", "-").replace("–", "-")
        equation = equation.replace("×", "*").replace("·", "*")
        equation = equation.replace("\\left", "").replace("\\right", "")
        equation = self._latex_commands.sub(r"\1", equation)
        equation = self._spaces.sub("", equation)
        return equation


def build_model() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "char_ngrams",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 7),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_boundary_ngrams",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 6),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline(
        [
            ("normalize", EquationNormalizer()),
            ("features", features),
            ("classifier", classifier),
        ]
    )
