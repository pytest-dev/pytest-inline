from inline import itest

a = 0
a = a + 1
itest("1", tag = ["add"]).given(a, 1).check_eq(a, 2)
itest("1", tag = ["add"]).given(a, 1).check_eq(a, 2)
itest("1", tag = ["add"]).given(a, 1).check_eq(a, 2)
a = a + 2
itest("2").given(a, 1).check_eq(a, 3)
itest("2").given(a, 1).check_eq(a, 3)
itest("2").given(a, 1).check_eq(a, 3)
a = a - 1
itest("3", tag = ["minus"]).given(a, 1).check_eq(a, 0)
itest("3", tag = ["minus"]).given(a, 1).check_eq(a, 0)
itest("3", tag = ["minus"]).given(a, 1).check_eq(a, 0)
