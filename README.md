# pytest-inline

pytest-inline is a plugin for [pytest](<http://pytest.org>) that writes inline tests.


## Table of contents

1. [Example](#Example)
2. [Install](#Install)
3. [Use](#Use)
4. [API](#API)
5. [Citation](#Citation)

## Example

```python {.line-numbers}
def FileHeader(self):
        dt = self.date_time
        dosdate = (dt[0] - 1980) << 9 | dt[1] << 5 | dt[2]
        Here().given(dt, (1980, 1, 25, 17, 13, 14)).check_eq(dosdate, 57)
        dostime = dt[3] << 11 | dt[4] << 5 | (dt[5] // 2)
        Here().given(dt, (1980, 1, 25, 17, 13, 14)).check_eq(dostime, 35239)
        if self.flag_bits & 0x08:
                # Set these to zero because we write them after the file data
                CRC = compress_size = file_size = 0
```

## Install

Use ``pip install pytest-inline`` to install this plugin

## Use

Use ``pytest .`` to run all inline tests in working directory 

Use ``pytest {filename}`` to run all inline tests in a Python file

## API

### Declaration an inline test

- Here(test_name, parameterized, repeated, tag, disabled, timeout): 
1. test_name is a string that represents the name of the test. The default value is the file name + line number of the test statement.

2. parameterized is a boolean value that indicates whether the test is parameterized. The default value is false.

3. repeated is an integer that indicates how many times the test is repeated. The default value is 1.
        
4. tag is a string that represents the tag of the test. The default value is an empty string.
        
5. disabled is a boolean value that indicates whether the test is disabled. The default value is false.
        
6. timeout is an integer that represents the timeout of the test. The default value is -1.


### Provide test inputs

- given(variable, value): 
        assign the value to the variable.


### Specify test oracles

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

[paper-url]: /README.md
