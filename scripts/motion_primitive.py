from enum import Enum


class MotionPrimitive():
    """
    #WIP
    A motion primitive that defines a trajectory from a over a time T
    """

    def __init__(self, constructor_type, constructor_data):
        """
        Input:
            constructor_type,      How the motion primitive will be constructed (e.g. from a polynomial, from a series of constant jerks), of type MotionPrimitiveConstructorType
            constructor_data,      The data corresponding to the input_type (e.g. a list of polynomial coefficients, a list of jerks with switching times)
        Usage example:
            MotionPrimitive(MotionPrimitiveConstructorType.POLYNOMIAL,[1,2,3,4,5])
        """

        self.constructor = constructor_type
        self.constructor(constructor_data)

    @staticmethod
    def polynomial_constructor(constructor_data):
        """
        constructor_data = polynomial coefficients
        """
        pass

    @staticmethod
    def jerks_constructor(constructor_data):
        """
        constructor_data = ([switch times],[jerk values]) 
        """
        pass

    def get_state(self, t):
        """
        Given a time t, return the state of the motion primitive at that time
        """
        pass


class MotionPrimitiveConstructorType(Enum):
    POLYNOMIAL = MotionPrimitive.polynomial_constructor
    JERKS = MotionPrimitive.jerks_constructor


if __name__ == "__main__":
    mp = MotionPrimitive(MotionPrimitiveConstructorType.POLYNOMIAL, [1, 2, 3, 4, 5])
