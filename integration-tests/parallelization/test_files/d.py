from inline import Here
import time

sleep = 1
d = 0
d = d + 1
Here("1", tag = ["add"]).given(d, 1).check_eq(d, 2).check_eq(time.sleep(sleep), None)
Here("1", tag = ["add"]).given(d, 1).check_eq(d, 2)
Here("1", tag = ["add"]).given(d, 1).check_eq(d, 2)
d = d + 2
Here("2").given(d, 1).check_eq(d, 3).check_eq(time.sleep(sleep), None)
Here("2").given(d, 1).check_eq(d, 3)
Here("2").given(d, 1).check_eq(d, 3)
d = d - 1
Here("3", tag = ["minus"]).given(d, 1).check_eq(d, 0).check_eq(time.sleep(sleep), None)
Here("3", tag = ["minus"]).given(d, 1).check_eq(d, 0)
Here("3", tag = ["minus"]).given(d, 1).check_eq(d, 0)