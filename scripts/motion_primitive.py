from enum import Enum


class MotionPrimitive():
    """
    #WIP
    A motion primitive that defines a trajectory from a over a time T. Put functions that all MPs should have in here
    """

    def __init__(self):
        """
        """
        pass

    def get_state(self, t):
        """
        Given a time t, return the state of the motion primitive at that time. Hopefully agnostic to subclass(or have a function named this in both subclasses)
        """
        pass


class PolynomialMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from polynomial coefficients
    """

    def __init__(self, polynomial_coeffs):
        """
        """
        self.polynomial_constructor(polynomial_coeffs)

    def polynomial_constructor(self, polynomial_coeffs):
        """
        """
        pass


class JerksMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from a sequence of constant jerks
    """

    def __init__(self, jerks_data):
        self.jerks_constructor(jerks_data)

    def jerks_constructor(self, jerks_data):
        """
        jerks_data = ([switch times],[jerk values]) 
        """
        pass


if __name__ == "__main__":
    mp = PolynomialMotionPrimitive([1, 2, 3, 4, 5])
