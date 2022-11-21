from inline import Here
import time

sleep = 2
i = 0
i = i + 1
Here("1", tag = ["add"]).given(i, 1).check_eq(i, 2).check_eq(time.sleep(sleep), None)
Here("1", tag = ["add"]).given(i, 1).check_eq(i, 2)
Here("1", tag = ["add"]).given(i, 1).check_eq(i, 2)
i = i + 2
Here("2").given(i, 1).check_eq(i, 3).check_eq(time.sleep(sleep), None)
Here("2").given(i, 1).check_eq(i, 3)
Here("2").given(i, 1).check_eq(i, 3)
i = i - 1
Here("3", tag = ["minus"]).given(i, 1).check_eq(i, 0).check_eq(time.sleep(sleep), None)
Here("3", tag = ["minus"]).given(i, 1).check_eq(i, 0)
Here("3", tag = ["minus"]).given(i, 1).check_eq(i, 0)