from inline import Here
import time

sleep = 2
g = 0
g = g + 1
Here("1", tag = ["add"]).given(g, 1).check_eq(g, 2).check_eq(time.sleep(sleep), None)
Here("1", tag = ["add"]).given(g, 1).check_eq(g, 2)
Here("1", tag = ["add"]).given(g, 1).check_eq(g, 2)
g = g + 2
Here("2").given(g, 1).check_eq(g, 3).check_eq(time.sleep(sleep), None)
Here("2").given(g, 1).check_eq(g, 3)
Here("2").given(g, 1).check_eq(g, 3)
g = g - 1
Here("3", tag = ["minus"]).given(g, 1).check_eq(g, 0).check_eq(time.sleep(sleep), None)
Here("3", tag = ["minus"]).given(g, 1).check_eq(g, 0)
Here("3", tag = ["minus"]).given(g, 1).check_eq(g, 0)