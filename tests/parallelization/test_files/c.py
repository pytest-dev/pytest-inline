from inline import Here
import time

sleep = 2
c = 0
c = c + 1
Here("1", tag = ["add"]).given(c, 1).check_eq(c, 2).check_eq(time.sleep(sleep), None)
Here("1", tag = ["add"]).given(c, 1).check_eq(c, 2)
Here("1", tag = ["add"]).given(c, 1).check_eq(c, 2)
c = c + 2
Here("2").given(c, 1).check_eq(c, 3).check_eq(time.sleep(sleep), None)
Here("2").given(c, 1).check_eq(c, 3)
Here("2").given(c, 1).check_eq(c, 3)
c = c - 1
Here("3", tag = ["minus"]).given(c, 1).check_eq(c, 0).check_eq(time.sleep(sleep), None)
Here("3", tag = ["minus"]).given(c, 1)
Here("3", tag = ["minus"]).given(c, 1)