from inline import itest
import time

sleep = 1
d = 0
d = d + 1
itest("1", tag = ["add"]).given(d, 1).check_eq(d, 2).check_eq(time.sleep(sleep), None)
itest("1", tag = ["add"]).given(d, 1).check_eq(d, 2)
itest("1", tag = ["add"]).given(d, 1).check_eq(d, 2)
d = d + 2
itest("2").given(d, 1).check_eq(d, 3).check_eq(time.sleep(sleep), None)
itest("2").given(d, 1).check_eq(d, 3)
itest("2").given(d, 1).check_eq(d, 3)
d = d - 1
itest("3", tag = ["minus"]).given(d, 1).check_eq(d, 0).check_eq(time.sleep(sleep), None)
itest("3", tag = ["minus"]).given(d, 1).check_eq(d, 0)
itest("3", tag = ["minus"]).given(d, 1).check_eq(d, 0)