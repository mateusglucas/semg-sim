import numpy as np

class CurrentTripole:
    def __init__(self, P1, d12, d13):
        self.P1 = P1
        self.d12 = d12
        self.d13 = d13

    @property
    def P2(self):
        return self._calculate_P2_and_P3()[0]

    @property
    def P3(self):
        return self._calculate_P2_and_P3()[1]

    def _calculate_P2_and_P3(self):
        # encontrar valores de P2 e P3. P2+P3=-P1 e P2*d12+P3*d13=0
        # A*x=y, com A=[1, 1; d12, d13], x=[P2; P3], e y=[-P1; 0], logo [P2; P3]=A\b;
        A=np.array([[1,1],[self.d12, self.d13]])
        y=np.array([-self.P1, 0])

        return np.linalg.solve(A, y)