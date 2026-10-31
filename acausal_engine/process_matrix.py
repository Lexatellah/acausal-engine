"""
Core algebra for process matrices in the Choi representation.
"""

import numpy as np


class ChoiMatrix:
    """A wrapper around a Choi matrix with validation utilities."""

    def __init__(self, matrix: np.ndarray, tol: float = 1e-9):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Choi matrix must be square.")
        self.matrix = matrix
        self.tol = tol
        self._validate()

    def _validate(self):
        if not np.allclose(self.matrix, self.matrix.conj().T, atol=self.tol):
            raise ValueError("Choi matrix is not Hermitian.")
        eigvals = np.linalg.eigvalsh(self.matrix)
        if np.any(eigvals < -self.tol):
            raise ValueError(
                f"Choi matrix is not positive semidefinite. "
                f"Min eigenvalue: {eigvals.min():.3e}"
            )

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def dim(self) -> int:
        return self.matrix.shape[0]

    def trace(self) -> complex:
        return np.trace(self.matrix)

    def normalize(self) -> "ChoiMatrix":
        tr = self.trace()
        if abs(tr) < self.tol:
            raise ValueError("Cannot normalize zero-trace matrix.")
        return ChoiMatrix(self.matrix / tr, tol=self.tol)

    def __repr__(self) -> str:
        return f"ChoiMatrix(dim={self.dim}, trace={self.trace():.4f})"
