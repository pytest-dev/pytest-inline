from inline import Here
import time

sleep = 2
e = 0
e = e + 1
Here("1", tag = ["add"]).given(e, 1).check_eq(e, 2).check_eq(time.sleep(sleep), None)
Here("1", tag = ["add"]).given(e, 1).check_eq(e, 2)
Here("1", tag = ["add"]).given(e, 1).check_eq(e, 2)
e = e + 2
Here("2").given(e, 1).check_eq(e, 3).check_eq(time.sleep(sleep), None)
Here("2").given(e, 1).check_eq(e, 3)
Here("2").given(e, 1).check_eq(e, 3)
e = e - 1
Here("3", tag = ["minus"]).given(e, 1).check_eq(e, 0).check_eq(time.sleep(sleep), None)
Here("3", tag = ["minus"]).given(e, 1).check_eq(e, 0)
Here("3", tag = ["minus"]).given(e, 1).check_eq(e, 0)