# pytest-inline

![License](https://img.shields.io/github/license/EngineeringSoftware/pytest-inline)
[![Pypi](https://img.shields.io/pypi/v/pytest-inline)](https://pypi.org/project/pytest-inline/)
[![Release](https://img.shields.io/github/v/release/EngineeringSoftware/pytest-inline?include_prereleases)](https://github.com/EngineeringSoftware/pytest-inline/releases)
[![GithubWorkflow](https://img.shields.io/github/actions/workflow/status/EngineeringSoftware/pytest-inline/python-package.yml?branch=main)](https://github.com/EngineeringSoftware/pytest-inline/actions/workflows/python-package.yml)

pytest-inline is a [pytest](<http://pytest.org>) plugin for writing inline tests.

Inline testing is a new granularity of testing that makes it easier to check individual program statements. An inline test is a statement that allows developers to provide arbitrary inputs and test oracles for checking the immediately preceding statement that is not an inline test.  Unlike unit tests that are usually placed in separate `test_*.py` files, inline tests are written together with the actual production code (and thus are easier to maintain). 

## Table of contents

1. [Example](#Example)
2. [Install](#Install)
3. [Use](#Use)
4. [API](#API)
5. [Performance](#Performance)
6. [Citation](#Citation)

## Example
The regular expression (Line 7) in this code snippet checks if variable name matches a regex for a pattern that ends in a colon and has at least one digit
The inline test (Line 8) that we write for target statement (Line 7) consists of three parts:
- Declaration with itest() constructor
- Assigning inputs with given() function calls
- Specifying test oracles with check_*() function calls

```python
from inline import itest

def get_assignment_map_from_checkpoint(tvars, init_c):
    ...
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        itest().given(name, "a:0").check_eq(m, "a")
        if m is not None:
            name = m.group(1)
    ...
```

## Install

Use ``pip install pytest-inline`` to install this plugin.

## Use

Use ``pytest .`` to run all inline tests in working directory.

Use ``pytest {filename}`` to run all inline tests in a Python file.

## API

### Declaration of an inline test

- itest(test_name, parameterized, repeated, tag, disabled, timeout): 
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
        itest().assume(2 < 4).given(dt, (1980, 1, 25, 17, 13, 14)).check_eq(dosdate, 57)
    ```



### Provide test inputs using given calls

- given(variable, value): 
        Assign the value to the variable. 

    Note that any number of given statements can be added. Below is a small example of this functionality. Additionally, the first given call must proceed either an itest() declaration or an assume() call if it is added.

    ```python {.line-numbers}
    def multiple_givens(a, c):
        b = a + c
        itest().given(a, 2).given(c, a + 1).check_true(b == 5)
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
        

## Performance

Inline tests are generally fast to run, as each inline test only checks one statement.  Note that inline tests behave as empty function calls when not running tests, and the overhead of having them in production code is negligible.

We have evaluated the performance of pytest-inline on a [dataset](https://github.com/EngineeringSoftware/inlinetest/tree/main/data/examples/python) of 87 inline tests we wrote for 80 statements in 50 examples from 31 open-source projects.  The main findings are summarized below, and please see [our paper][paper-url] for more details.  We performed 3 experiments:

1. Running inline tests in standalone mode.  Each inline test took 0.147s on average.  Most time is spent on startup and parsing the file, which can be aromatized if there are more inline tests per file.  As such, we also duplicated the inline tests for 10, 100, 1000 times, and the average time per inline test dropped to 0.015s, 0.002s, 0.001s.

2. Running inline tests together with unit tests. The average overhead compared to running only the unit tests was only 0.007x.  Even when duplicating the inline tests for 1000 times (such that the total number of inline tests match the total number of unit tests), the average overhead was only 0.088x.

3. Running inline tests together with unit tests, but disable pytest-inline.  This is to simulate the cost of having inline tests in production code.  The overhead was negligible: the number we got when not duplicating inline tests was -0.001x (due to noise), and only raised to 0.019x when duplicating the inline tests for 1000 times.

All APIs for inline testing behave as empty function calls in non-testing mode by always returning a dummy object, for example, check\_eq is defined as: `def check_eq(self, ...): return self`.  This usually incurs negligible overhead as we have observed in our experiments, but note that the cost is paid each time an inline test is encountered during execution, so it may add up if the inline test is in a part of code that will be executed many times (e.g., a loop).


## Citation

Title: [Inline Tests](https://dl.acm.org/doi/abs/10.1145/3551349.3556952)

Authors: [Yu Liu](https://sweetstreet.github.io/), [Pengyu Nie](https://pengyunie.github.io/), [Owolabi Legunsen](https://mir.cs.illinois.edu/legunsen/), [Milos Gligoric](http://users.ece.utexas.edu/~gligoric/)

```bibtex
@inproceedings{LiuASE22InlineTests,
  title =        {Inline Tests},
  author =       {Yu Liu and Pengyu Nie and Owolabi Legunsen and Milos Gligoric},
  pages =        {1--13},
  booktitle =    {International Conference on Automated Software Engineering},
  year =         {2022},
}
```

Title: [pytest-inline](https://pengyunie.github.io/p/LiuETAL23pytest-inline.pdf)

Authors: [Yu Liu](https://sweetstreet.github.io/), [Zachary Thurston](), [Alan Han](), [Pengyu Nie](https://pengyunie.github.io/), [Milos Gligoric](http://users.ece.utexas.edu/~gligoric/), [Owolabi Legunsen](https://mir.cs.illinois.edu/legunsen/)

```bibtex
@inproceedings{LiuICSE23PytestInline,
  title =        {pytest-inline: An Inline Testing Tool for Python},
  author =       {Yu Liu and Zachary Thurston and Alan Han and Pengyu Nie and Milos Gligoric and Owolabi Legunsen},
  pages =        {1--4},
  booktitle =    {International Conference on Software Engineering, DEMO},
  year =         {2023},
}
```
