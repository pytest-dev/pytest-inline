from inline import itest
import time

sleep = 1
h = 0
h = h + 1
itest("1", tag = ["add"]).given(h, 1).check_eq(h, 2).check_eq(time.sleep(sleep), None)
itest("1", tag = ["add"]).given(h, 1).check_eq(h, 2)
itest("1", tag = ["add"]).given(h, 1).check_eq(h, 2)
h = h + 2
itest("2").given(h, 1).check_eq(h, 3).check_eq(time.sleep(sleep), None)
itest("2").given(h, 1).check_eq(h, 3)
itest("2").given(h, 1).check_eq(h, 3)
h = h - 1
itest("3", tag = ["minus"]).given(h, 1).check_eq(h, 0).check_eq(time.sleep(sleep), None)
itest("3", tag = ["minus"]).given(h, 1).check_eq(h, 0)
itest("3", tag = ["minus"]).given(h, 1).check_eq(h, 0)