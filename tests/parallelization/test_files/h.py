from inline import Here
import time

sleep = 1
h = 0
h = h + 1
Here("1", tag = ["add"]).given(h, 1).check_eq(h, 2).check_eq(time.sleep(sleep), None)
Here("1", tag = ["add"]).given(h, 1).check_eq(h, 2)
Here("1", tag = ["add"]).given(h, 1).check_eq(h, 2)
h = h + 2
Here("2").given(h, 1).check_eq(h, 3).check_eq(time.sleep(sleep), None)
Here("2").given(h, 1).check_eq(h, 3)
Here("2").given(h, 1).check_eq(h, 3)
h = h - 1
Here("3", tag = ["minus"]).given(h, 1).check_eq(h, 0).check_eq(time.sleep(sleep), None)
Here("3", tag = ["minus"]).given(h, 1).check_eq(h, 0)
Here("3", tag = ["minus"]).given(h, 1).check_eq(h, 0)