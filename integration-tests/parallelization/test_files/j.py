from inline import itest
import time

sleep = 1
j = 0
j = j + 1
itest("1", tag = ["add"]).given(j, 1).check_eq(j, 2).check_eq(time.sleep(sleep), None)
itest("1", tag = ["add"]).given(j, 1).check_eq(j, 2)
itest("1", tag = ["add"]).given(j, 1).check_eq(j, 2)
j = j + 2
itest("2").given(j, 1).check_eq(j, 3).check_eq(time.sleep(sleep), None)
itest("2").given(j, 1).check_eq(j, 3)
itest("2").given(j, 1).check_eq(j, 3)
j = j - 1
itest("3", tag = ["minus"]).given(j, 1).check_eq(j, 0).check_eq(time.sleep(sleep), None)
itest("3", tag = ["minus"]).given(j, 1).check_eq(j, 0)
itest("3", tag = ["minus"]).given(j, 1).check_eq(j, 0)