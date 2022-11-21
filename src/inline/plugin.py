import ast
import copy
import inspect
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import enum
import pytest
from _pytest.pathlib import fnmatch_ex, import_path
from pytest import Collector, Config, FixtureRequest, Parser
import signal
from _pytest.main import Session

if sys.version_info >= (3, 9, 0):
    from ast import unparse as ast_unparse
else:
    from .ast_future import unparse as ast_unparse


# Alternatively, invoke pytest with -p inline)
# pytest_plugins = ["inline"]

# register argparse-style options and ini-file values, called once at the beginning of a test run
def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("collect")
    group.addoption(
        "--inlinetest-only",
        action="store_true",
        default=False,
        help="run inlinetests in all .py modules",
        dest="inlinetest_only",
    )
    group.addoption(
        "--inlinetest-glob",
        action="append",
        default=[],
        metavar="pat",
        help="inlinetests file matching pattern, default: *.py",
        dest="inlinetest_glob",
    )
    group.addoption(
        "--inlinetest-continue-on-failure",
        action="store_true",
        default=False,
        help="for a given inlinetest, continue to run after the first failure",
        dest="inlinetest_continue_on_failure",
    )
    group.addoption(
        "--inlinetest-ignore-import-errors",
        action="store_true",
        default=False,
        help="ignore inlinetest ImportErrors",
        dest="inlinetest_ignore_import_errors",
    )
    group.addoption(
        "--inlinetest-disable",
        action="store_true",
        default=False,
        help="disable inlinetests",
        dest="inlinetest_disable",
    )
    group.addoption(
        "--inlinetest-group",
        action="append",
        default=[],
        metavar="tag",
        help="group inlinetests",
        dest="inlinetest_group",
    )
    group.addoption(
        "--inlinetest-order",
        action="append",
        default=[],
        metavar="tag",
        help="order inlinetests",
        dest="inlinetest_order",
    )


@pytest.hookimpl()
def pytest_exception_interact(node, call, report):
    if isinstance(call.excinfo.value, MalformedException) or isinstance(
        call.excinfo.value, AssertionError
    ):
        # fail to parse inline test
        # fail to execute inline test
        entry = report.longrepr.reprtraceback.reprentries[-1]
        entry.style = "short"
        entry.lines = entry.lines[-3:]
        report.longrepr.reprtraceback.reprentries = [entry]


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line("markers", "inline: mark inline tests")


@pytest.hookimpl()
def pytest_collectstart(collector):
    if not isinstance(collector, Session):
        if collector.config.getoption("inlinetest_only") and (
            not isinstance(collector, InlinetestModule)
        ):
            collector.collect = lambda: []  # type: ignore[assignment]
        elif collector.config.getoption("inlinetest_disable") and isinstance(
            collector, InlinetestModule
        ):
            collector.collect = lambda: []  # type: ignore[assignment]


def pytest_collect_file(
    file_path: Path,
    parent: Collector,
) -> Optional["InlinetestModule"]:
    config = parent.config
    if _is_inlinetest(config, file_path):
        mod: InlinetestModule = InlinetestModule.from_parent(parent, path=file_path)
        return mod
    return None


def _is_inlinetest(config: Config, file_path: Path) -> bool:
    if config.getoption("inlinetest_disable"):
        return False
    globs = config.getoption("inlinetest_glob") or ["*.py"]
    return any(fnmatch_ex(glob, file_path) for glob in globs)


@pytest.fixture(scope="session")
def inlinetest_namespace() -> Dict[str, Any]:
    """Fixture that returns a :py:class:`dict` that will be injected into the
    namespace of inlinetests."""
    return dict()


######################################################################
## InlineTest
######################################################################
class InlineTest:
    # https://docs.python.org/3/tutorial/stdlib.html
    import_libraries = [
        "import re",
        "import unittest",
        "from unittest.mock import patch",
    ]

    def __init__(self):
        self.assume_stmts = []
        self.assume_node : ast.If = None
        self.check_stmts = []
        self.given_stmts = []
        self.previous_stmts = []
        self.prev_stmt_type = PrevStmtType.StmtExpr
        # the line number of test statement
        self.lineno = 0
        self.test_name = ""
        # flag of parameterized test
        self.parameterized = False
        # parameterized inline tests List[InlineTest]
        self.parameterized_inline_tests = []
        self.repeated = 1
        self.tag = []
        self.disabled = False
        self.timeout = -1.0
        self.globs = {}

    def to_test(self):
        if self.prev_stmt_type == PrevStmtType.CondExpr:
            if self.assume_stmts == []: 
                return "\n".join(
                    self.import_libraries
                    + [ExtractInlineTest.node_to_source_code(n) for n in self.given_stmts]
                    + [ExtractInlineTest.node_to_source_code(n) for n in self.check_stmts]
                )
            else:
                body_nodes = [n for n in self.given_stmts] + [n for n in self.previous_stmts] + [n for n in self.check_stmts]
                assume_statement = self.assume_stmts[0]
                assume_node = self.build_assume_node(assume_statement, body_nodes)
                return "\n".join(
                    self.import_libraries
                    + ExtractInlineTest.node_to_source_code(assume_node)
                    
                )
                

        else:
            if self.assume_stmts == []:
                return "\n".join(
                    self.import_libraries
                    + [ExtractInlineTest.node_to_source_code(n) for n in self.given_stmts]
                    + [ExtractInlineTest.node_to_source_code(n) for n in self.previous_stmts]
                    + [ExtractInlineTest.node_to_source_code(n) for n in self.check_stmts]
                )
            else:
                body_nodes = [n for n in self.given_stmts] + [n for n in self.previous_stmts] + [n for n in self.check_stmts]
                assume_statement = self.assume_stmts[0]
                assume_node = self.build_assume_node(assume_statement, body_nodes)
                return "\n".join(
                    self.import_libraries
                    + [ExtractInlineTest.node_to_source_code(assume_node)]
                )
    
    def build_assume_node(self, assumption_node, body_nodes):
        return ast.If(assumption_node, body_nodes,[])

    def __repr__(self):
        if self.test_name:
            return f"inline test {self.test_name}, starting at line {self.lineno}"
        else:
            return f"inline test, starting at line {self.lineno}"

    def is_empty(self) -> bool:
        return not self.check_stmts

    def __eq__(self, other):
        return (
            self.import_libraries == other.import_libraries
            and self.assume_stmts == other.assume_stmts
            and self.given_stmts == other.given_stmts
            and self.previous_stmts == other.previous_stmts
            and self.check_stmts == other.check_stmts
        )


class PrevStmtType(enum.Enum):
    # the previous statement is a statement expression
    StmtExpr = 1
    # the previous statement is a conditional expression
    CondExpr = 2


class MalformedException(Exception):
    """
    An invalid inline test
    """

    pass


class TimeoutException(Exception):
    """
    Time limit exceeded
    """

    pass


######################################################################
## InlineTest Parser
######################################################################
class InlinetestParser:
    def parse(self, obj, globs: None):
        # obj = open(self.file_path, "r").read():
        if isinstance(obj, ModuleType):
            tree = ast.parse(open(obj.__file__, "r").read())
        else:
            return []

        # bind node's parent and children
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
                if isinstance(child, ast.stmt):
                    node.children = (
                        [child]
                        if not hasattr(node, "children")
                        else [child] + node.children
                    )

        extract_inline_test = ExtractInlineTest()
        extract_inline_test.visit(tree)
        if globs:
            for inline_test in extract_inline_test.inline_test_list:
                inline_test.globs = copy.copy(globs)
        return extract_inline_test.inline_test_list


class ExtractInlineTest(ast.NodeTransformer):
    class_name_str = "Here"
    check_eq_str = "check_eq"
    check_true_str = "check_true"
    check_false_str = "check_false"
    check_none_str = "check_none"
    check_not_none_str = "check_not_none"
    check_neq_str = "check_neq"
    check_same = "check_same"
    check_not_same = "check_not_same"
    fail_str = "fail"
    given_str = "given"
    group_str = "Group"
    arg_test_name_str = "test_name"
    arg_parameterized_str = "parameterized"
    arg_repeated_str = "repeated"
    arg_tag_str = "tag"
    arg_disabled_str = "disabled"
    arg_timeout_str = "timeout"
    assume = "assume"
    inline_module_imported = False

    def __init__(self):
        self.cur_inline_test = InlineTest()
        self.inline_test_list = []

    def is_inline_test_class(self, node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == self.class_name_str:
                return True
            elif isinstance(node.func, ast.Attribute):
                # e.g. print(ast.dump(ast.parse('snake.colour', mode='eval'), indent=4))
                # snake is Attribute and colour is Name
                return self.is_inline_test_class(node.func.value)
            else:
                return False
        else:
            return False

    def visit_ImportFrom(self, node):
        if node.module == "inline" and node.names[0].name == "Here":
            self.inline_module_imported = True
        return self.generic_visit(node)

    def find_condition_stmt(self, stmt_node):
        if isinstance(stmt_node.parent, ast.If) or isinstance(
            stmt_node.parent, ast.While
        ):
            self.cur_inline_test.prev_stmt_type = PrevStmtType.CondExpr
            return stmt_node.parent.test
        else:
            raise NotImplementedError("inline test: failed to find a tested statement")

    def find_previous_stmt(self, node):
        # get the previous stmt that is not Here() by finding the previous sibling
        stmt_node = node
        while not isinstance(stmt_node, ast.Expr):
            stmt_node = stmt_node.parent
        index_stmt_node = stmt_node.parent.children.index(stmt_node)
        if index_stmt_node < 0 or index_stmt_node > len(stmt_node.parent.children) - 1:
            raise MalformedException(
                f"inline test: failed to find statement {ExtractInlineTest.node_to_source_code(stmt_node)} in {self.class_name_str}"
            )
        elif index_stmt_node == len(stmt_node.parent.children) - 1:
            # the first stmt in the block, but AST is parsed in revserse order so it appears to be the last one in children list
            # if / while block
            return self.find_condition_stmt(stmt_node)
        else:
            for i in range(1, len(stmt_node.parent.children) - index_stmt_node):
                prev_stmt_node = stmt_node.parent.children[index_stmt_node + i]
                if isinstance(
                    prev_stmt_node.value, ast.Call
                ) and self.is_inline_test_class(prev_stmt_node.value):
                    continue
                else:
                    return prev_stmt_node
            return self.find_condition_stmt(prev_stmt_node)

    def collect_inline_test_calls(self, node, inline_test_calls: List[ast.Call]):
        """
        collect all function calls in the node
        """
        if isinstance(node, ast.Attribute):
            self.collect_inline_test_calls(node.value, inline_test_calls)
        elif isinstance(node, ast.Call):
            inline_test_calls.append(node)
            self.collect_inline_test_calls(node.func, inline_test_calls)

    def parse_constructor(self, node):
        """
        Parse a constructor call.
        """
        NUM_OF_ARGUMENTS = 6
        if len(node.args) + len(node.keywords) <= NUM_OF_ARGUMENTS:
            # positional arguments
            if sys.version_info >= (3, 8, 0):
                for index, arg in enumerate(node.args):
                    # check if "test_name" is a string
                    if (
                        index == 0
                        and isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                    ):
                        # get the test name if exists
                        self.cur_inline_test.test_name = arg.value
                    # check if "parameterized" is a boolean
                    elif (
                        index == 1
                        and isinstance(arg, ast.Constant)
                        and isinstance(arg.value, bool)
                    ):
                        self.cur_inline_test.parameterized = arg.value
                    # check if "repeated" is a positive integer
                    elif (
                        index == 2
                        and isinstance(arg, ast.Constant)
                        and isinstance(arg.value, int)
                    ):
                        if arg.value <= 0:
                            raise MalformedException(
                                f"inline test: {self.arg_repeated_str} must be greater than 0"
                            )
                        self.cur_inline_test.repeated = arg.value
                    elif index == 3 and isinstance(arg.value, ast.List):
                        tags = []
                        for elt in arg.value.elts:
                            if not (
                                isinstance(elt, ast.Constant)
                                and isinstance(elt.value, str)
                            ):
                                raise MalformedException(
                                    f"tag can only be List of string"
                                )
                            tags.append(elt.value)
                        self.cur_inline_test.tag = tags
                    elif (
                        index == 4
                        and isinstance(arg, ast.Constant)
                        and isinstance(arg.value, bool)
                    ):
                        self.cur_inline_test.disabled = arg.value
                    elif (
                        index == 5
                        and isinstance(arg, ast.Constant)
                        and isinstance(arg.value, float)
                    ):
                        self.cur_inline_test.timeout = arg.value
                    else:
                        raise MalformedException(
                            f"inline test: Here() accepts {NUM_OF_ARGUMENTS} arguments. 'test_name' must be a string constant, 'parameterized' must be a boolean constant, 'repeated' must be a positive integer, 'tag' must be a list of string, 'timeout' must be a positive float"
                        )
                # keyword arguments
                for keyword in node.keywords:
                    # check if "test_name" is a string
                    if (
                        keyword.arg == self.arg_test_name_str
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        self.cur_inline_test.test_name = keyword.value.value
                    # check if "parameterized" is a boolean
                    elif (
                        keyword.arg == self.arg_parameterized_str
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, bool)
                    ):
                        self.cur_inline_test.parameterized = keyword.value.value
                    # check if "repeated" is a positive integer
                    elif (
                        keyword.arg == self.arg_repeated_str
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, int)
                    ):
                        if keyword.value.value <= 0:
                            raise MalformedException(
                                f"inline test: {self.arg_repeated_str} must be greater than 0"
                            )
                        self.cur_inline_test.repeated = keyword.value.value
                    # check if "tag" is a list of string
                    elif keyword.arg == self.arg_tag_str and isinstance(
                        keyword.value, ast.List
                    ):
                        tags = []
                        for elt in keyword.value.elts:
                            if not (
                                isinstance(elt, ast.Constant)
                                and isinstance(elt.value, str)
                            ):
                                raise MalformedException(
                                    f"tag can only be List of string"
                                )
                            tags.append(elt.value)
                        self.cur_inline_test.tag = tags
                    # check if "disabled" is a boolean
                    elif (
                        keyword.arg == self.arg_disabled_str
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, bool)
                    ):
                        self.cur_inline_test.disabled = keyword.value.value
                    # check if "timeout" is a positive float
                    elif (
                        keyword.arg == self.arg_timeout_str
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, float)
                    ):
                        if keyword.value.value <= 0.0:
                            raise MalformedException(
                                f"inline test: {self.arg_timeout_str} must be greater than 0"
                            )
                        self.cur_inline_test.timeout = keyword.value.value
                    else:
                        raise MalformedException(
                            f"inline test: Here() accepts {NUM_OF_ARGUMENTS} arguments. 'test_name' must be a string constant, 'parameterized' must be a boolean constant, 'repeated' must be a positive integer, 'tag' must be a list of string, 'timeout' must be a positive float"
                        )
            else:
                for index, arg in enumerate(node.args):
                    # check if "test_name" is a string
                    if (
                        index == 0
                        and isinstance(arg, ast.Str)
                        and isinstance(arg.s, str)
                    ):
                        # get the test name if exists
                        self.cur_inline_test.test_name = arg.s
                    # check if "parameterized" is a boolean
                    elif (
                        index == 1
                        and isinstance(arg, ast.NameConstant)
                        and isinstance(arg.value, bool)
                    ):
                        self.cur_inline_test.parameterized = arg.value
                    # check if "repeated" is a positive integer
                    elif (
                        index == 2
                        and isinstance(arg, ast.Num)
                        and isinstance(arg.n, int)
                    ):
                        if arg.n <= 0.0:
                            raise MalformedException(
                                f"inline test: {self.arg_repeated_str} must be greater than 0"
                            )
                        self.cur_inline_test.repeated = arg.n
                    # check if "tag" is a list of string
                    elif index == 3 and isinstance(arg.value, ast.List):
                        tags = []
                        for elt in arg.value.elts:
                            if not (
                                isinstance(elt, ast.Str) and isinstance(elt.s, str)
                            ):
                                raise MalformedException(
                                    f"tag can only be List of string"
                                )
                            tags.append(elt.s)
                        self.cur_inline_test.tag = tags
                    # check if "disabled" is a boolean
                    elif (
                        index == 4
                        and isinstance(arg, ast.NameConstant)
                        and isinstance(arg.value, bool)
                    ):
                        self.cur_inline_test.disabled = arg.value
                    # check if "timeout" is a positive int
                    elif (
                        index == 5
                        and isinstance(arg, ast.Num)
                        and isinstance(arg.n, float)
                    ):
                        if arg.n <= 0.0:
                            raise MalformedException(
                                f"inline test: {self.arg_timeout_str} must be greater than 0"
                            )
                        self.cur_inline_test.timeout = arg.n
                    else:
                        raise MalformedException(
                            f"inline test: Here() accepts {NUM_OF_ARGUMENTS} arguments. 'test_name' must be a string constant, 'parameterized' must be a boolean constant, 'repeated' must be a positive intege, 'tag' must be a list of string, 'timeout' must be a positive float"
                        )
                # keyword arguments
                for keyword in node.keywords:
                    # check if "test_name" is a string
                    if (
                        keyword.arg == self.arg_test_name_str
                        and isinstance(keyword.value, ast.Str)
                        and isinstance(keyword.value.s, str)
                    ):
                        self.cur_inline_test.test_name = keyword.value.s
                    # check if "parameterized" is a boolean
                    elif (
                        keyword.arg == self.arg_parameterized_str
                        and isinstance(keyword.value, ast.NameConstant)
                        and isinstance(keyword.value.value, bool)
                    ):
                        self.cur_inline_test.parameterized = keyword.value.value
                    # check if "repeated" is a positive integer
                    elif (
                        keyword.arg == self.arg_repeated_str
                        and isinstance(keyword.value, ast.Num)
                        and isinstance(keyword.value.n, int)
                    ):
                        if keyword.value.n <= 0.0:
                            raise MalformedException(
                                f"inline test: {self.arg_repeated_str} must be greater than 0"
                            )
                        self.cur_inline_test.repeated = keyword.value.n
                    # check if "tag" is a list of string
                    elif keyword.arg == self.arg_tag_str and isinstance(
                        keyword.value, ast.List
                    ):
                        tags = []
                        for elt in keyword.value.elts:
                            if not (
                                isinstance(elt, ast.Str) and isinstance(elt.s, str)
                            ):
                                raise MalformedException(
                                    f"tag can only be List of string"
                                )
                            tags.append(elt.s)
                        self.cur_inline_test.tag = tags
                    # check if "disabled" is a boolean
                    elif (
                        keyword.arg == self.arg_disabled_str
                        and isinstance(keyword.value, ast.NameConstant)
                        and isinstance(keyword.value.value, bool)
                    ):
                        self.cur_inline_test.disabled = keyword.value.value
                    # check if "timeout" is a positive float
                    elif (
                        keyword.arg == self.arg_timeout_str
                        and isinstance(keyword.value, ast.Num)
                        and isinstance(keyword.value.n, float)
                    ):
                        if keyword.value.n <= 0.0:
                            raise MalformedException(
                                f"inline test: {self.arg_timeout_str} must be greater than 0"
                            )
                        self.cur_inline_test.timeout = keyword.value.n
                    else:
                        raise MalformedException(
                            f"inline test: Here() accepts {NUM_OF_ARGUMENTS} arguments. 'test_name' must be a string constant, 'parameterized' must be a boolean constant, 'repeated' must be a positive integer, 'tag' must be a list of string, 'timeout' must be a positive float"
                        )
        else:
            raise MalformedException(
                "inline test: invalid Here(), expected at most 3 args"
            )

        if not self.cur_inline_test.test_name:
            # by default, use lineno as test name
            self.cur_inline_test.test_name = f"line{node.lineno}"
        # set the line number
        self.cur_inline_test.lineno = node.lineno

    def parameterized_inline_tests_init(self, node: ast.List):
        if not self.cur_inline_test.parameterized_inline_tests:
            self.cur_inline_test.parameterized_inline_tests = [
                InlineTest() for _ in range(len(node.elts))
            ]
        if len(node.elts) != len(self.cur_inline_test.parameterized_inline_tests):
            raise MalformedException(
                "inline test: parameterized tests must have the same number of test cases"
            )

    def parse_given(self, node):
        if len(node.args) == 2:
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[1])
                for index, value in enumerate(node.args[1].elts):
                    assign_node = ast.Assign(targets=[node.args[0]], value=value)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].given_stmts.append(assign_node)
            else:
                assign_node = ast.Assign(targets=[node.args[0]], value=node.args[1])
                self.cur_inline_test.given_stmts.append(assign_node)
        else:
            raise MalformedException("inline test: invalid given(), expected 2 args")

    def parse_assume(self, node):
        if len(node.args) == 1:
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[0])
                for index, value in enumerate(node.args[0].elts):
                    test_node = self.parse_group(value)
                    assumption_node = self.build_assume(test_node)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].assume_stmts.append(assumption_node)
            else:
                test_node = self.parse_group(node.args[0])
                self.cur_inline_test.assume_stmts.append(test_node)
        else:
            raise MalformedException(
                "inline test: invalid assume() call, expected 1 arg"
            )

    def build_assert_eq(self, left_node, comparator_node):
        equal_node = ast.Compare(
            left=left_node,
            ops=[ast.Eq()],
            comparators=[comparator_node],
        )
        assert_node = ast.Assert(
            test=equal_node,
            msg=ast.Call(
                func=ast.Attribute(
                    ast.Constant("{0} == {1}\nActual: {2}\nExpected: {3}\n"),
                    "format",
                    ast.Load(),
                ),
                args=[
                    ast.Constant(self.node_to_source_code(left_node)),
                    ast.Constant(self.node_to_source_code(comparator_node)),
                    left_node,
                    comparator_node,
                ],
                keywords=[],
            ),
        )
        return assert_node

    def parse_check_eq(self, node):
        # check if the function being called is an inline test function
        if len(node.args) == 2:
            left_node = self.parse_group(node.args[0])
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[1])
                for index, value in enumerate(node.args[1].elts):
                    comparator_node = self.parse_group(value)
                    assert_node = self.build_assert_eq(left_node, comparator_node)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].check_stmts.append(assert_node)
            else:
                comparator_node = self.parse_group(node.args[1])
                assert_node = self.build_assert_eq(left_node, comparator_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException("inline test: invalid check_eq(), expected 2 args")

    def build_assert_true(self, test_node):
        assert_node = ast.Assert(
            test=test_node,
            msg=ast.Call(
                func=ast.Attribute(
                    ast.Constant(
                        "bool({0}) is True\nActual: bool({1}) is False\nExpected: bool({1}) is True\n"
                    ),
                    "format",
                    ast.Load(),
                ),
                args=[
                    ast.Constant(self.node_to_source_code(test_node)),
                    test_node,
                ],
                keywords=[],
            ),
        )
        return assert_node

    def parse_check_true(self, node):
        if len(node.args) == 1:
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[0])
                for index, value in enumerate(node.args[0].elts):
                    test_node = self.parse_group(value)
                    assert_node = self.build_assert_true(test_node)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].check_stmts.append(assert_node)
            else:
                test_node = self.parse_group(node.args[0])
                assert_node = self.build_assert_true(test_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException(
                "inline test: invalid check_true(), expected 1 arg"
            )

    def build_assert_false(self, operand_node):
        assert_node = ast.Assert(
            test=ast.UnaryOp(op=ast.Not(), operand=operand_node),
            msg=ast.Call(
                func=ast.Attribute(
                    ast.Constant(
                        "bool({0}) is False\nActual: bool({1}) is True\nExpected: bool({1}) is False\n"
                    ),
                    "format",
                    ast.Load(),
                ),
                args=[
                    ast.Constant(self.node_to_source_code(operand_node)),
                    operand_node,
                ],
                keywords=[],
            ),
        )
        return assert_node

    def parse_check_false(self, node):
        if len(node.args) == 1:
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[0])
                for index, value in enumerate(node.args[0].elts):
                    operand_node = self.parse_group(value)
                    assert_node = self.build_assert_false(operand_node)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].check_stmts.append(assert_node)
            else:
                operand_node = self.parse_group(node.args[0])
                assert_node = self.build_assert_false(operand_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException(
                "inline test: invalid check_false(), expected 1 arg"
            )

    def build_assert_neq(self, left_node, comparator_node):
        equal_node = ast.Compare(
            left=left_node,
            ops=[ast.NotEq()],
            comparators=[comparator_node],
        )
        assert_node = ast.Assert(
            test=equal_node,
            msg=ast.Call(
                func=ast.Attribute(
                    ast.Constant("{0} == {1}\nActual: {2}\nExpected: {3}\n"),
                    "format",
                    ast.Load(),
                ),
                args=[
                    ast.Constant(self.node_to_source_code(left_node)),
                    ast.Constant(self.node_to_source_code(comparator_node)),
                    left_node,
                    comparator_node,
                ],
                keywords=[],
            ),
        )
        return assert_node

    def parse_check_neq(self, node):
        # check if the function being called is an inline test function
        if len(node.args) == 2:
            left_node = self.parse_group(node.args[0])
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[1])
                for index, value in enumerate(node.args[1].elts):
                    comparator_node = self.parse_group(value)
                    assert_node = self.build_assert_neq(left_node, comparator_node)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].check_stmts.append(assert_node)
            else:
                comparator_node = self.parse_group(node.args[1])
                assert_node = self.build_assert_neq(left_node, comparator_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException(
                "inline test: invalid check_neq(), expected 2 args"
            )

    def build_assert_none(self, left_node):
        equal_node = ast.Compare(
            left=left_node,
            ops=[ast.Is()],
            comparators=[ast.Constant(None)],
        )
        assert_node = ast.Assert(
            test=equal_node,
            msg=ast.Call(
                func=ast.Attribute(
                    ast.Constant("Assertion that value was None failed\n"),
                    "format",
                    ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
        )
        return assert_node

    def parse_check_none(self, node):
        if len(node.args) == 1:
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[0])
                for index, value in enumerate(node.args[0].elts):
                    operand_node = self.parse_group(value)
                    assert_node = self.build_assert_none(operand_node)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].check_stmts.append(assert_node)
            else:
                operand_node = self.parse_group(node.args[0])
                assert_node = self.build_assert_none(operand_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException(
                "inline test: invalid check_none(), expected 1 arg"
            )

    def build_assert_not_none(self, left_node):
        equal_node = ast.Compare(
            left=left_node,
            ops=[ast.IsNot()],
            comparators=[ast.Constant(None)],
        )
        assert_node = ast.Assert(
            test=equal_node,
            msg=ast.Call(
                func=ast.Attribute(
                    ast.Constant("Assertion that value was not None failed\n"),
                    "format",
                    ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
        )
        return assert_node

    def parse_check_not_none(self, node):
        if len(node.args) == 1:
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[0])
                for index, value in enumerate(node.args[0].elts):
                    operand_node = self.parse_group(value)
                    assert_node = self.build_assert_not_none(operand_node)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].check_stmts.append(assert_node)
            else:
                operand_node = self.parse_group(node.args[0])
                assert_node = self.build_assert_not_none(operand_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException(
                "inline test: invalid check_not_none(), expected 1 arg"
            )

    def build_assert_same(self, left_node, comparator_node):
        equal_node = ast.Compare(
            left=left_node,
            ops=[ast.Is()],
            comparators=[comparator_node],
        )
        assert_node = ast.Assert(
            test=equal_node,
            msg=ast.Call(
                func=ast.Attribute(
                    ast.Constant("{0} == {1}\nActual: {2}\nExpected: {3}\n"),
                    "format",
                    ast.Load(),
                ),
                args=[
                    ast.Constant(self.node_to_source_code(left_node)),
                    ast.Constant(self.node_to_source_code(comparator_node)),
                    left_node,
                    comparator_node,
                ],
                keywords=[],
            ),
        )
        return assert_node

    def parse_check_same(self, node):
        # check if the function being called is an inline test function
        if len(node.args) == 2:
            left_node = self.parse_group(node.args[0])
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[1])
                for index, value in enumerate(node.args[1].elts):
                    comparator_node = self.parse_group(value)
                    assert_node = self.build_assert_same(left_node, comparator_node)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].check_stmts.append(assert_node)
            else:
                comparator_node = self.parse_group(node.args[1])
                assert_node = self.build_assert_same(left_node, comparator_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException(
                "inline test: invalid check_same(), expected 2 args"
            )

    def build_assert_not_same(self, left_node, comparator_node):
        equal_node = ast.Compare(
            left=left_node,
            ops=[ast.IsNot()],
            comparators=[comparator_node],
        )
        assert_node = ast.Assert(
            test=equal_node,
            msg=ast.Call(
                func=ast.Attribute(
                    ast.Constant("{0} == {1}\nActual: {2}\nExpected: {3}\n"),
                    "format",
                    ast.Load(),
                ),
                args=[
                    ast.Constant(self.node_to_source_code(left_node)),
                    ast.Constant(self.node_to_source_code(comparator_node)),
                    left_node,
                    comparator_node,
                ],
                keywords=[],
            ),
        )
        return assert_node

    def parse_check_not_same(self, node):
        # check if the function being called is an inline test function
        if len(node.args) == 2:
            left_node = self.parse_group(node.args[0])
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[1])
                for index, value in enumerate(node.args[1].elts):
                    comparator_node = self.parse_group(value)
                    assert_node = self.build_assert_not_same(left_node, comparator_node)
                    self.cur_inline_test.parameterized_inline_tests[
                        index
                    ].check_stmts.append(assert_node)
            else:
                comparator_node = self.parse_group(node.args[1])
                assert_node = self.build_assert_not_same(left_node, comparator_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException(
                "inline test: invalid check_not_same(), expected 2 args"
            )

    def build_fail(self):
        equal_node = ast.Compare(
            left=ast.Constant(0),
            ops=[ast.Eq()],
            comparators=[ast.Constant(1)],
        )
        assert_node = ast.Assert(test=equal_node)
        return assert_node

    def parse_fail(self, node):
        # check if the function being called is an inline test function
        if len(node.args) == 0:
            self.build_fail()
        else:
            raise MalformedException("inline test: fail() does not expect any arguments")

    def parse_group(self, node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == self.group_str
        ):
            # node type is ast.Call, node.func type is ast.Name
            if sys.version_info >= (3, 8, 0):
                index_args = [arg.value for arg in node.args]
            else:
                # python3.7 type of arg is ast.Num which does not in higher version
                index_args = [arg.n for arg in node.args]
            if self.cur_inline_test.prev_stmt_type != PrevStmtType.CondExpr:
                raise MalformedException(
                    "inline test: Group() must be called to test a conditional statement"
                )
            if not self.cur_inline_test.previous_stmts:
                raise MalformedException("inline test: previous statement not found")
            stmt = self.cur_inline_test.previous_stmts[0]
            for i, index_arg in enumerate(index_args):
                if isinstance(stmt, ast.BoolOp):
                    if index_arg < 0 or index_arg >= len(stmt.values):
                        raise MalformedException(
                            f"inline test: Group() {i} index with value {index_arg} out of range"
                        )
                    else:
                        stmt = stmt.values[index_arg]
                        # raise NotImplementedError(index_arg, ast.dump(stmt))
            return stmt
        else:
            return node

    def parse_parameterized_test(self):
        for index, parameterized_test in enumerate(
            self.cur_inline_test.parameterized_inline_tests
        ):
            parameterized_test.previous_stmts = self.cur_inline_test.previous_stmts
            parameterized_test.prev_stmt_type = self.cur_inline_test.prev_stmt_type
            parameterized_test.lineno = self.cur_inline_test.lineno
            parameterized_test.test_name = (
                self.cur_inline_test.test_name + "_" + str(index)
            )

    def parse_inline_test(self, node):
        inline_test_calls = []
        self.collect_inline_test_calls(node, inline_test_calls)
        inline_test_calls.reverse()

        if len(inline_test_calls) <= 1:
            raise MalformedException(
                "inline test: invalid inline test, requires at least one assertion"
            )

        # "Here()" or "Here('test name')" or "Here('test name', True)" or "Here(parameterized=True)" or "Here(test_name='test name', parameterized=True)"
        constructor_call = inline_test_calls[0]
        if (
            isinstance(constructor_call.func, ast.Name)
            and constructor_call.func.id == self.class_name_str
        ):
            self.parse_constructor(constructor_call)
        else:
            raise MalformedException("inline test: invalid inline test constructor")

        # "assume_true(...) or assume_false(...)
        inline_test_call_index = 1
        if(len(inline_test_calls) >= 2):
            call = inline_test_calls[1]
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == self.assume
            ):
                self.parse_assume(call)
                inline_test_call_index += 1
        
        # "given(a, 1)"
        for call in inline_test_calls[inline_test_call_index:]:
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == self.given_str
            ):
                self.parse_given(call)
                inline_test_call_index += 1
            else:
                break

        # "check_eq" or "check_true" or "check_false" or "check_neq"
        for call in inline_test_calls[inline_test_call_index:]:
            # "check_eq(a, 1)"
            if call.func.attr == self.check_eq_str:
                self.parse_check_eq(call)
            # "check_true(a)"
            elif call.func.attr == self.check_true_str:
                self.parse_check_true(call)
            # "check_false(a)"
            elif call.func.attr == self.check_false_str:
                self.parse_check_false(call)
            # "check_neq(a, 1)"
            elif call.func.attr == self.check_neq_str:
                self.parse_check_neq(call)
            elif call.func.attr == self.check_none_str:
                self.parse_check_none(call)
            elif call.func.attr == self.check_not_none_str:
                self.parse_check_not_none(call)
            elif call.func.attr == self.check_same:
                self.parse_check_same(call)
            elif call.func.attr == self.check_not_same:
                self.parse_check_not_same(call)
            elif call.func.attr == self.fail_str:
                self.parse_fail(call)
            elif call.func.attr == self.given_str:
                raise MalformedException(
                    f"inline test: given() must be called before check_eq()/check_true()/check_false()"
                )
            else:
                raise MalformedException(
                    f"inline test: invalid function call {self.node_to_source_code(call.func)}"
                )

        if self.cur_inline_test.parameterized:
            self.parse_parameterized_test()
            self.inline_test_list.extend(
                self.cur_inline_test.parameterized_inline_tests
            )
        else:
            # add current inline test to the list
            self.inline_test_list.append(self.cur_inline_test)
        # init a new inline test object
        self.cur_inline_test = InlineTest()

    def visit_Expr(self, node):
        if self.inline_module_imported == False:
            return self.generic_visit(node)
        if self.is_inline_test_class(node.value):
            # get previous stmt
            self.cur_inline_test.previous_stmts.append(self.find_previous_stmt(node))
            # pase inline test
            self.parse_inline_test(node.value)
        return self.generic_visit(node)

    @staticmethod
    def node_to_source_code(node):
        ast.fix_missing_locations(node)
        return ast_unparse(node)

######################################################################
## InlineTest Finder
######################################################################
class InlineTestFinder:
    def __init__(self, parser=InlinetestParser(), recurse=True, exclude_empty=True):
        self._parser = parser
        self._recurse = recurse
        self._exclude_empty = exclude_empty

    def _from_module(self, module, object):
        """
        Return true if the given object is defined in the given
        module.
        """
        if module is None:
            return True
        elif inspect.getmodule(object) is not None:
            return module is inspect.getmodule(object)
        elif inspect.isfunction(object):
            return module.__dict__ is object.__globals__
        elif inspect.ismethoddescriptor(object):
            if hasattr(object, "__objclass__"):
                obj_mod = object.__objclass__.__module__
            elif hasattr(object, "__module__"):
                obj_mod = object.__module__
            else:
                return True  # [XX] no easy way to tell otherwise
            return module.__name__ == obj_mod
        elif inspect.isclass(object):
            return module.__name__ == object.__module__
        elif hasattr(object, "__module__"):
            return module.__name__ == object.__module__
        elif isinstance(object, property):
            return True  # [XX] no way not be sure.
        else:
            raise ValueError("object must be a class or function")

    def _is_routine(self, obj):
        """
        Safely unwrap objects and determine if they are functions.
        """
        maybe_routine = obj
        try:
            maybe_routine = inspect.unwrap(maybe_routine)
        except ValueError:
            pass
        return inspect.isroutine(maybe_routine)

    def find(self, obj, module=None, globs=None, extraglobs=None):
        # Find the module that contains the given object (if obj is
        # a module, then module=obj.).
        if module is False:
            module = None
        elif module is None:
            module = inspect.getmodule(obj)

        # Initialize globals, and merge in extraglobs.
        if globs is None:
            if module is None:
                globs = {}
            else:
                globs = module.__dict__.copy()
        else:
            globs = globs.copy()
        if extraglobs is not None:
            globs.update(extraglobs)
        if "__name__" not in globs:
            globs["__name__"] = "__main__"  # provide a default module name

        # Recursively explore `obj`, extracting InlineTests.
        tests = []
        self._find(tests, obj, module, globs, {})
        return tests

    def _find(self, tests, obj, module, globs, seen):
        if id(obj) in seen:
            return
        seen[id(obj)] = 1
        # Find a test for this object, and add it to the list of tests.
        test = self._parser.parse(obj, globs)
        if test is not None:
            tests.append(test)

        if inspect.ismodule(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                valname = "%s" % (valname)

                # Recurse to functions & classes.
                if (
                    self._is_routine(val) or inspect.isclass(val)
                ) and self._from_module(module, val):
                    self._find(tests, val, module, globs, seen)

        # Look for tests in a class's contained objects.
        if inspect.isclass(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                # Special handling for staticmethod/classmethod.
                if isinstance(val, (staticmethod, classmethod)):
                    val = val.__func__

                # Recurse to methods, properties, and nested classes.
                if (
                    inspect.isroutine(val)
                    or inspect.isclass(val)
                    or isinstance(val, property)
                ) and self._from_module(module, val):
                    valname = "%s" % (valname)
                    self._find(tests, val, module, globs, seen)


######################################################################
## InlineTest Runner
######################################################################
class InlineTestRunner:
    def run(self, test: InlineTest, out: List) -> None:
        tree = ast.parse(test.to_test())
        codeobj = compile(tree, filename="<ast>", mode="exec")
        start_time = time.time()
        if test.timeout > 0:
            with timeout(seconds=test.timeout):
                exec(codeobj, test.globs)
        else:
            exec(codeobj, test.globs)
        end_time = time.time()
        out.append(f"Test Execution time: {round(end_time - start_time, 4)} seconds")
        if test.globs:
            test.globs.clear()


class InlinetestItem(pytest.Item):
    def __init__(
        self,
        name: str,
        parent: "InlinetestModule",
        runner: Optional["InlineTestRunner"] = None,
        dtest: Optional["InlineTest"] = None,
    ) -> None:
        super().__init__(name, parent)
        self.runner = runner
        self.dtest = dtest
        self.obj = None
        self.fixture_request: Optional[FixtureRequest] = None
        self.add_marker(pytest.mark.inline)

    @classmethod
    def from_parent(
        cls,
        parent: "InlinetestModule",
        *,
        name: str,
        runner: "InlineTestRunner",
        dtest: "InlineTest",
    ):
        # incompatible signature due to imposed limits on subclass
        """The public named constructor."""
        return super().from_parent(name=name, parent=parent, runner=runner, dtest=dtest)

    def setup(self) -> None:
        if self.dtest is not None:
            self.fixture_request = _setup_fixtures(self)
            globs = dict(getfixture=self.fixture_request.getfixturevalue)
            for name, value in self.fixture_request.getfixturevalue(
                "inlinetest_namespace"
            ).items():
                globs[name] = value
            self.dtest.globs.update(globs)

    def runtest(self) -> None:
        assert self.dtest is not None
        assert self.runner is not None
        for round_index in range(1, self.dtest.repeated + 1):
            failures: List[str] = []
            self.runner.run(copy.copy(self.dtest), failures)
            if failures:
                print(failures)

    def reportinfo(self) -> Tuple[Union["os.PathLike[str]", str], Optional[int], str]:
        assert self.dtest is not None
        return self.path, self.dtest.lineno, "[inlinetest] %s" % self.name


class InlinetestModule(pytest.Module):
    def order_tests(test_list, tags):
        prio_unsorted = []
        unordered = []

        # sorting the tests based on if they are ordered or not
        for test in test_list:
            if len(set(test.tag) & set(tags)) > 0:
                prio_unsorted.append(test)
            else:
                unordered.append(test)

        # giving each test a value for its order in tags
        sorted_ordering = [None] * len(prio_unsorted)
        for i in range(0, len(prio_unsorted)):
            for tag in tags:
                if tag in prio_unsorted[i].tag:
                    sorted_ordering[i] = tags.index(tag)

        # sorting the list based on their tag positions
        prio_sorted = [
            val
            for (_, val) in sorted(
                zip(sorted_ordering, prio_unsorted), key=lambda x: x[0]
            )
        ]
        prio_sorted.extend(unordered)

        return prio_sorted

    def collect(self) -> Iterable[InlinetestItem]:
        if self.path.name == "conftest.py":
            module = self.config.pluginmanager._importconftest(
                self.path,
                self.config.getoption("importmode"),
                rootpath=self.config.rootpath,
            )
        else:
            try:
                # TODO: still need to find the right way to import without errors. mode=ImportMode.importlib did not work
                module = import_path(self.path, root=self.config.rootpath)
            except ImportError:
                if self.config.getvalue("inlinetest_ignore_import_errors"):
                    pytest.skip("unable to import module %r" % self.path)
                else:
                    raise ImportError("unable to import module %r" % self.path)
        finder = InlineTestFinder()
        runner = InlineTestRunner()

        group_tags = self.config.getoption("inlinetest_group", default=None)
        order_tags = self.config.getoption("inlinetest_order", default=None)

        for test_list in finder.find(module):
            # reorder the list if there are tests to be ordered
            ordered_list = InlinetestModule.order_tests(test_list, order_tags)
            if ordered_list is not None:
                for test in ordered_list:
                    if (
                        test.is_empty()
                        or (group_tags and len(set(test.tag) & set(group_tags)) == 0)
                        or test.disabled
                    ):  # skip empty inline tests and tests with tags not in the tag list and disabled tests
                        continue

                    yield InlinetestItem.from_parent(
                        self,
                        name=test.test_name,
                        runner=runner,
                        dtest=test,
                    )


def _setup_fixtures(inlinetest_item: InlinetestItem) -> FixtureRequest:
    """Used by InlinetestItem to setup fixture information."""

    def func() -> None:
        pass

    inlinetest_item.funcargs = {}  # type: ignore[attr-defined]
    fm = inlinetest_item.session._fixturemanager
    inlinetest_item._fixtureinfo = fm.getfixtureinfo(  # type: ignore[attr-defined]
        node=inlinetest_item, func=func, cls=None, funcargs=False
    )
    fixture_request = FixtureRequest(inlinetest_item, _ispytest=True)
    fixture_request._fillfixtures()
    return fixture_request


######################################################################################
#                                     Timeout                                        #
#                                     Logic                                          #
######################################################################################


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise (TimeoutException)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
