from inline import Here

a = 0
a = a + 1
Here("1", tag = ["add"]).given(a, 1).check_eq(a, 2)
Here("1", tag = ["add"]).given(a, 1).check_eq(a, 2)
Here("1", tag = ["add"]).given(a, 1).check_eq(a, 2)
a = a + 2
Here("2").given(a, 1).check_eq(a, 3)
Here("2").given(a, 1).check_eq(a, 3)
Here("2").given(a, 1).check_eq(a, 3)
a = a - 1
Here("3", tag = ["minus"]).given(a, 1).check_eq(a, 0)
Here("3", tag = ["minus"]).given(a, 1).check_eq(a, 0)
Here("3", tag = ["minus"]).given(a, 1).check_eq(a, 0)
