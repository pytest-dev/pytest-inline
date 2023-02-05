from inline import itest
import time

sleep = 1
f = 0
f = f + 1
itest("1", tag = ["add"]).given(f, 1).check_eq(f, 2).check_eq(time.sleep(sleep), None)
itest("1", tag = ["add"]).given(f, 1).check_eq(f, 2)
itest("1", tag = ["add"]).given(f, 1).check_eq(f, 2)
f = f + 2
itest("2").given(f, 1).check_eq(f, 3).check_eq(time.sleep(sleep), None)
itest("2").given(f, 1).check_eq(f, 3)
itest("2").given(f, 1).check_eq(f, 3)
f = f - 1
itest("3", tag = ["minus"]).given(f, 1).check_eq(f, 0).check_eq(time.sleep(sleep), None)
itest("3", tag = ["minus"]).given(f, 1).check_eq(f, 0)
itest("3", tag = ["minus"]).given(f, 1).check_eq(f, 0)