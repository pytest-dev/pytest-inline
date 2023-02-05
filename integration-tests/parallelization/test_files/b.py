from inline import itest
import time

sleep = 1
b = 0
b = b + 1
itest("1", tag = ["add"]).given(b, 1).check_eq(b, 2).check_eq(time.sleep(sleep), None)
itest("1", tag = ["add"]).given(b, 1).check_eq(b, 2)
itest("1", tag = ["add"]).given(b, 1).check_eq(b, 2)
b = b + 2
itest("2").given(b, 1).check_eq(b, 3).check_eq(time.sleep(sleep), None)
itest("2").given(b, 1).check_eq(b, 3)
itest("2").given(b, 1).check_eq(b, 3)
b = b - 1
itest("3", tag = ["minus"]).given(b, 1).check_eq(b, 0).check_eq(time.sleep(sleep), None)
itest("3", tag = ["minus"]).given(b, 1).check_eq(b, 0)
itest("3", tag = ["minus"]).given(b, 1).check_eq(b, 0)