from ast import List


class Here:
    def __init__(
        self,
        test_name: str = None,
        parameterized: bool = False,
        repeated: int = 1,
        tag: List = [],
        disabled: bool = False,
        timeout: float = -1.0,
    ):
        """
        Initialize Inline object with test name / parametrized flag

        :param test_name: test
        :param parameterized: whether the test is parameterized
        :param repeated: number of times to repeat the tests
        :param tag: tags to group tests
        :param disabled: whether the test is disabled
        :param timeout: seconds to timeout the test, must be a float
        """

    def given(self, variable, value):
        """
        Set value to a variable.

        :param variable: a variable name
        :param value: a value that will be assigned to the variable
        :returns: Inline object
        """
        return self

    def check_eq(self, actual_value, expected_value):
        """
        Assert whether two values equal

        :param actual_value: the value to check against expected
        :param expected_value: expected value
        :returns: Inline object
        :raises: AssertionError
        """
        return self

    def check_neq(self, actual_value, expected_value):
        """
        Assert whether two values are not equal

        :param actual_value: a value to check against expected
        :param expected_value: expected value
        :returns: Inline object
        :raises: AssertionError
        """
        return self

    def check_true(self, expr):
        """
        Assert whether a boolean expression is true

        :param expr: a boolean expression
        :returns: Inline object
        :raises: AssertionError
        """
        return self

    def check_false(self, expr):
        """
        Assert whether a boolean expression is false

        :param expr: a boolean expression
        :returns: Inline object
        :raises: AssertionError
        """
        return self

    def check_none(self, variable):
        """
        Assert whether a variable is None

        :param variable: a variable to check against
        :returns: Inline object
        :raises: AssertionError
        """
        return self

    def check_not_none(self, variable):
        """
        Assert whether a variable is not None

        :param variable: a variable to check against
        :returns: Inline object
        :raises: AssertionError
        """
        return self

    def check_same(self, actual_value, expected_value):
        """
        Assert whether an object is the same as a given expected object

        :param actual_value: a value to check against expected
        :param expected_value: expected value
        :returns: Inline object
        :raises: AssertionError
        """
        return self

    def check_not_same(self, actual_value, expected_value):
        """
        Assert whether an object is not the same as a given expected object

        :param actual_value: a value to check against expected
        :param expected_value: expected value
        :returns: Inline object
        :raises: AssertionError
        """
        return self

    def fail(self):
        """
        Fails the test
        
        :returns: Inline object
        :raises: AssertionError
        """
    
    def assume(self, condition:bool):
        """
        Executes the test under the assuming the given condition is true. If supplied, must be supplied immediately after Here().
        Can only supply 1 assume statement.

        :param condition: a boolean condition
        :raises: AssertionError
        """

class Group:
    def __init__(self, *arg):
        """
        Initialize Group object with index
        """
