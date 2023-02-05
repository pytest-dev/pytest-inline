from inline import itest
import time

sleep = 2
c = 0
c = c + 1
itest("1", tag = ["add"]).given(c, 1).check_eq(c, 2).check_eq(time.sleep(sleep), None)
itest("1", tag = ["add"]).given(c, 1).check_eq(c, 2)
itest("1", tag = ["add"]).given(c, 1).check_eq(c, 2)
c = c + 2
itest("2").given(c, 1).check_eq(c, 3).check_eq(time.sleep(sleep), None)
itest("2").given(c, 1).check_eq(c, 3)
itest("2").given(c, 1).check_eq(c, 3)
c = c - 1
itest("3", tag = ["minus"]).given(c, 1).check_eq(c, 0).check_eq(time.sleep(sleep), None)
itest("3", tag = ["minus"]).given(c, 1)
itest("3", tag = ["minus"]).given(c, 1)