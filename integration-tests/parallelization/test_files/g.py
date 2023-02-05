from inline import itest
import time

sleep = 2
g = 0
g = g + 1
itest("1", tag = ["add"]).given(g, 1).check_eq(g, 2).check_eq(time.sleep(sleep), None)
itest("1", tag = ["add"]).given(g, 1).check_eq(g, 2)
itest("1", tag = ["add"]).given(g, 1).check_eq(g, 2)
g = g + 2
itest("2").given(g, 1).check_eq(g, 3).check_eq(time.sleep(sleep), None)
itest("2").given(g, 1).check_eq(g, 3)
itest("2").given(g, 1).check_eq(g, 3)
g = g - 1
itest("3", tag = ["minus"]).given(g, 1).check_eq(g, 0).check_eq(time.sleep(sleep), None)
itest("3", tag = ["minus"]).given(g, 1).check_eq(g, 0)
itest("3", tag = ["minus"]).given(g, 1).check_eq(g, 0)