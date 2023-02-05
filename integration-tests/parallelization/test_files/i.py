from inline import itest
import time

sleep = 2
i = 0
i = i + 1
itest("1", tag = ["add"]).given(i, 1).check_eq(i, 2).check_eq(time.sleep(sleep), None)
itest("1", tag = ["add"]).given(i, 1).check_eq(i, 2)
itest("1", tag = ["add"]).given(i, 1).check_eq(i, 2)
i = i + 2
itest("2").given(i, 1).check_eq(i, 3).check_eq(time.sleep(sleep), None)
itest("2").given(i, 1).check_eq(i, 3)
itest("2").given(i, 1).check_eq(i, 3)
i = i - 1
itest("3", tag = ["minus"]).given(i, 1).check_eq(i, 0).check_eq(time.sleep(sleep), None)
itest("3", tag = ["minus"]).given(i, 1).check_eq(i, 0)
itest("3", tag = ["minus"]).given(i, 1).check_eq(i, 0)