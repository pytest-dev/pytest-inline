=============
pytest-inline
=============

pytest-inline is a plugin for `pytest <http://pytest.org>`_ that writes inline tests.

Installation and usage
----------------------

Use ``pip install pytest-inline`` to install our Python pytest-plugin

Use ``pytest .`` to run all inline tests in working directory 

Use ``pytest {filename}`` to run all inline tests in a Python file

API
---
Declaration an inline test

- Here(test_name, parameterized, repeated, tag, disabled, timeout): 
        1. test_name is a string that represents the name of the test. The default value is the file name + line number of the test statement.
        2. parameterized is a boolean value that indicates whether the test is parameterized. The default value is false.
        3. repeated is an integer that indicates how many times the test is repeated. The default value is 1.
        4. tag is a string that represents the tag of the test. The default value is an empty string.
        5. disabled is a boolean value that indicates whether the test is disabled. The default value is false.
        6. timeout is an integer that represents the timeout of the test. The default value is -1.


Provide test inputs

- given(variable, value): 
        assign the value to the variable.


Specify test oracles

- check\_eq(actual\_value, expected\_value): 
        check if the actual value is equal to the expected value.
- check\_neq(actual\_value, expected\_value): 
        check if the actual value is not equal to the expected value.
- check\_true(expr): 
        check if the boolean expression is true.
- check\_false(expr): 
        check if the boolean expression is false.
- check\_none(variable): 
        check if the variable is none.
- check\_not\_none(variable): 
        check if the variable is not none.
- check\_same(actual\_value, expected\_value): 
        check if the actual value and the expected value refer to the same object.
- check\_not\_same(actual\_value, expected\_value): 
        check if the actual value and the expected value refer to different objects.
