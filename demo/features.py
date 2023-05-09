import time
from inline import itest

### display name: customize test name, default name is filename+line number

def inline_test_with_name(a):
    b = a + 1
    itest(test_name="test-with-name").given(a, 1).check_eq(b, 2)


### parameterized tests: pass different sets of inputs to tests

def inline_test_parameterized(a):
    b = a + 1
    itest(parameterized=True).given(a, [1, 2, 3]).check_eq(b, [2, 3, 4])


### repeated tests: repeat a test a specified number of times

def inline_test_repeated(a):
    b = a + 1
    itest(repeated=3).given(a, 1).check_eq(b, 2)


### disabled tests: disable a test

def inline_test_disabled(a):
    b = a + 1
    itest(disabled=True).given(a, 1).check_eq(b, "this test is disabled")


### timeout: fail a test if the execution time exceeds a given duration

def slow_method():
    time.sleep(0.01)
    # time.sleep(0.1)
    return 1

def inline_test_with_timeout(a):
    b = a + 1
    # this inline test will fail if you increase the sleep time in slow_method
    itest(timeout=0.1, test_name="timeout-expected-to-fail").given(a, slow_method()).check_eq(b, 2)


### assumptions: execute test when the assumption is satisfied

def inline_test_with_assume(a):
    b = a + 1
    itest().assume(False).given(a, 1).check_eq(b, 2)


### more assertions

def inline_test_assertions(a):
    b = a + 1
    itest().given(a, 1).check_neq(b, 1)

    c = None if b == 2 else []
    itest().given(b, 2).check_none(c)
    itest().given(b, 1).check_not_none(c)

    d = c if b == 2 else list(c)
    itest().given(b, 2).given(c, []).check_same(c, d)
    itest().given(b, 1).given(c, []).check_not_same(c, d)


### tagged tests: tag tests for filtering
### run tests in order: Run some tests first

def inline_test_with_tags(a):
    b = a + 1
    itest(test_name="foo1", tag=["foo"]).given(a, 1).check_eq(b, 2)
    itest(test_name="foo2", tag=["foo"]).given(a, 2).check_eq(b, 3)
    itest(test_name="bar3", tag=["bar"]).given(a, 3).check_eq(b, 4)
    itest(test_name="bar4", tag=["bar"]).given(a, 4).check_eq(b, 5)
