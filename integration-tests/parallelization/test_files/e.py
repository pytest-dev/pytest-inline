from inline import itest
import time

sleep = 2
e = 0
e = e + 1
itest("1", tag = ["add"]).given(e, 1).check_eq(e, 2).check_eq(time.sleep(sleep), None)
itest("1", tag = ["add"]).given(e, 1).check_eq(e, 2)
itest("1", tag = ["add"]).given(e, 1).check_eq(e, 2)
e = e + 2
itest("2").given(e, 1).check_eq(e, 3).check_eq(time.sleep(sleep), None)
itest("2").given(e, 1).check_eq(e, 3)
itest("2").given(e, 1).check_eq(e, 3)
e = e - 1
itest("3", tag = ["minus"]).given(e, 1).check_eq(e, 0).check_eq(time.sleep(sleep), None)
itest("3", tag = ["minus"]).given(e, 1).check_eq(e, 0)
itest("3", tag = ["minus"]).given(e, 1).check_eq(e, 0)