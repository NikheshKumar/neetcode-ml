import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.asarray(x, dtype=np.float64)
        W1 = np.asarray(W1, dtype=np.float64)
        W2 = np.asarray(W2, dtype=np.float64)
        b1 = np.asarray(b1, dtype=np.float64)
        b2 = np.asarray(b2, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        z1 = np.dot(W1,x) + b1
        a1 = np.maximum(0.0, z1)
        y_pred = np.dot(W2, a1) + b2

        l = np.mean((y_pred-y_true)**2)
        dz2 = 2.0 * (y_pred - y_true) / len(y_true)
        dW2 = np.outer(dz2, a1) 
        db2 = dz2
        da1 = W2.T @ dz2
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {"loss":np.round(float(l), 4), "dW1":np.round(dW1, 4), "db1":np.round(db1, 4), "dW2":np.round(dW2, 4), "db2":np.round(db2, 4)}




