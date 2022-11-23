# pytest-inline

pytest-inline is a plugin for [pytest](<http://pytest.org>) that writes inline tests.

Inline testing is a new granularity of testing that make it easier to check individual program statements. An inline test is a statement that allows to provide arbitrary inputs and test oracles for checking the immediately preceding statement that is not an inline test.


## Table of contents

1. [Example](#Example)
2. [Install](#Install)
3. [Use](#Use)
4. [API](#API)
5. [Citation](#Citation)

## Example
The regular expression (Line 5) in this code snippet checks if variable name matches a regex for a pattern that ends in a colon and has at least one digit
The inline test (Line 6) that we write for target statement (Line 5) consists of three parts:
- Declaration with Here() constructor
- Assigning inputs with given() function calls
- Specifying test oracles with check_*() function calls

```python
def get_assignment_map_from_checkpoint(tvars, init_c):
    ...
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        Here().given(name, "a:0").check_eq(m, "a")
        if m is not None:
            name = m.group(1)
    ...
```

## Install

Use ``pip install pytest-inline`` to install this plugin.

## Use

Use ``pytest .`` to run all inline tests in working directory .

Use ``pytest {filename}`` to run all inline tests in a Python file.

## API

### Declaration of an inline test

- Here(test_name, parameterized, repeated, tag, disabled, timeout): 
1. test_name is a string that represents the name of the test. The default value is the file name + line number of the test statement.

2. parameterized is a boolean value that indicates whether the test is parameterized. The default value is false.

3. repeated is an integer that indicates how many times the test is repeated. The default value is 1.
        
4. tag is a string that represents the tag of the test. The default value is an empty string.
        
5. disabled is a boolean value that indicates whether the test is disabled. The default value is false.
        
6. timeout is an float that represents the amount of time to run the test until it will timeout. The default value is -1.0. If a test times out, it will throw a timeout exception.

### Provide any assumptions using an assume call

- assume(condition):
        Checks if the condition holds, if the condition does not hold the test will not be run. 
        
    If an assume check is added, it must occur before any given statements and only one can be added.

    Below is a toy example. It merely illustrates how a call would look with an assume statement. More complex logic can be used in the assume statment, such as checking a given system version and assuming that the test should only run if the assumed system version is true.

    ```python {.line-numbers}
    def FileHeader(self):
            dt = self.date_time
            dosdate = (dt[0] - 1980) << 9 | dt[1] << 5 | dt[2]
            Here().assume(2 < 4).given(dt, (1980, 1, 25, 17, 13, 14)).check_eq(dosdate, 57)
    ```



### Provide test inputs using given calls

- given(variable, value): 
        Assign the value to the variable. 

    Note that any number of given statements can be added. Below is a small example of this functionality. Additionally, the first given call must proceed either a Here() declaration or a assume() call if it is added.

    ```python {.line-numbers}
    def multiple_givens(a, c):
        b = a + c
        Here().given(a, 2).given(c, a + 1).check_true(b == 5)
    ```


### Specify test oracles using check calls
- check\_eq(actual\_value, expected\_value): 
        Checks if the actual value is equal to the expected value.

- check\_neq(actual\_value, expected\_value): 
        Checks if the actual value is not equal to the expected value.

- check\_true(expr): 
        Checks if the boolean expression is true.

- check\_false(expr): 
        Checks if the boolean expression is false.

- check\_none(variable): 
        Checks if the variable is none.

- check\_not\_none(variable): 
        Checks if the variable is not none.

- check\_same(actual\_value, expected\_value): 
        Checks if the actual value and the expected value refer to the same object.

- check\_not\_same(actual\_value, expected\_value): 
        Checks if the actual value and the expected value refer to different objects.

Only one test oracle can be specified for a given inline test call.
        

# Citation

Title: [Inline Tests][paper-url]

Authors: [Yu Liu](https://sweetstreet.github.io/), [Pengyu Nie](https://pengyunie.github.io/), [Owolabi Legunsen](https://mir.cs.illinois.edu/legunsen/), [Milos Gligoric](http://users.ece.utexas.edu/~gligoric/)

If you have used I-Test in a research project, please cite the research paper in any related publication:

```bibtex
@inproceedings{LiuASE22InlineTests,
  title =        {Inline Tests},
  author =       {Yu Liu and Pengyu Nie and Owolabi Legunsen and Milos Gligoric},
  pages =        {to appear},
  booktitle =    {International Conference on Automated Software Engineering},
  year =         {2022},
}
```

[paper-url]: https://arxiv.org/abs/2209.06315
