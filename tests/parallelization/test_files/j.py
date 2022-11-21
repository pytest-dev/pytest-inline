from inline import Here
import time

sleep = 1
j = 0
j = j + 1
Here("1", tag = ["add"]).given(j, 1).check_eq(j, 2).check_eq(time.sleep(sleep), None)
Here("1", tag = ["add"]).given(j, 1).check_eq(j, 2)
Here("1", tag = ["add"]).given(j, 1).check_eq(j, 2)
j = j + 2
Here("2").given(j, 1).check_eq(j, 3).check_eq(time.sleep(sleep), None)
Here("2").given(j, 1).check_eq(j, 3)
Here("2").given(j, 1).check_eq(j, 3)
j = j - 1
Here("3", tag = ["minus"]).given(j, 1).check_eq(j, 0).check_eq(time.sleep(sleep), None)
Here("3", tag = ["minus"]).given(j, 1).check_eq(j, 0)
Here("3", tag = ["minus"]).given(j, 1).check_eq(j, 0)