from inline import Here
import time

sleep = 1
b = 0
b = b + 1
Here("1", tag = ["add"]).given(b, 1).check_eq(b, 2).check_eq(time.sleep(sleep), None)
Here("1", tag = ["add"]).given(b, 1).check_eq(b, 2)
Here("1", tag = ["add"]).given(b, 1).check_eq(b, 2)
b = b + 2
Here("2").given(b, 1).check_eq(b, 3).check_eq(time.sleep(sleep), None)
Here("2").given(b, 1).check_eq(b, 3)
Here("2").given(b, 1).check_eq(b, 3)
b = b - 1
Here("3", tag = ["minus"]).given(b, 1).check_eq(b, 0).check_eq(time.sleep(sleep), None)
Here("3", tag = ["minus"]).given(b, 1).check_eq(b, 0)
Here("3", tag = ["minus"]).given(b, 1).check_eq(b, 0)