from inline import Here
import time

sleep = 1
f = 0
f = f + 1
Here("1", tag = ["add"]).given(f, 1).check_eq(f, 2).check_eq(time.sleep(sleep), None)
Here("1", tag = ["add"]).given(f, 1).check_eq(f, 2)
Here("1", tag = ["add"]).given(f, 1).check_eq(f, 2)
f = f + 2
Here("2").given(f, 1).check_eq(f, 3).check_eq(time.sleep(sleep), None)
Here("2").given(f, 1).check_eq(f, 3)
Here("2").given(f, 1).check_eq(f, 3)
f = f - 1
Here("3", tag = ["minus"]).given(f, 1).check_eq(f, 0).check_eq(time.sleep(sleep), None)
Here("3", tag = ["minus"]).given(f, 1).check_eq(f, 0)
Here("3", tag = ["minus"]).given(f, 1).check_eq(f, 0)