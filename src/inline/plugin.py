import ast
import copy
import enum
import inspect
import signal
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pytest
from _pytest.pathlib import fnmatch_ex
from pytest import Collector, Config, Parser

if sys.version_info >= (3, 9, 0):
    from ast import unparse as ast_unparse
else:
    from .ast_future import unparse as ast_unparse

if pytest.version_tuple >= (8, 0, 0):
    # fixture API changed in pytest 8
    # https://github.com/pytest-dev/pytest/issues/11218
    from _pytest.fixtures import TopRequest  # noqa: I001

    # consider_namespace_packages is added as a required argument in pytest 8
    # https://github.com/pytest-dev/pytest/issues/11475
    from _pytest.pathlib import import_path as _import_path  # noqa: I001

    def import_path(*args, **kwargs):
        return _import_path(*args, **kwargs, consider_namespace_packages=False)

    # scope architecture changed in pytest 8
    # https://github.com/pytest-dev/pytest/issues/7777
    from _pytest.main import Session, Dir  # noqa: I001
    from _pytest.python import Package  # noqa: I001

    HIGHLEVEL_SCOPES = (Session, Dir, Package)

else:
    from pytest import FixtureRequest  # noqa: I001
    from _pytest.pathlib import import_path  # noqa: I001
    from _pytest.main import Session  # noqa: I001
    from _pytest.python import Package  # noqa: I001

    HIGHLEVEL_SCOPES = (Session, Package)


# register argparse-style options and ini-file values, called once at the beginning of a test run
def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--inlinetest-only",
        action="store_true",
        default=False,
        help="run inlinetests in all .py modules",
        dest="inlinetest_only",
    )
    parser.addoption(
        "--inlinetest-glob",
        action="append",
        default=[],
        metavar="pat",
        help="inlinetests file matching pattern, default: *.py",
        dest="inlinetest_glob",
    )
    parser.addoption(
        "--inlinetest-continue-on-failure",
        action="store_true",
        default=False,
        help="for a given inlinetest, continue to run after the first failure",
        dest="inlinetest_continue_on_failure",
    )
    parser.addoption(
        "--inlinetest-ignore-import-errors",
        action="store_true",
        default=False,
        help="ignore inlinetest ImportErrors",
        dest="inlinetest_ignore_import_errors",
    )
    parser.addoption(
        "--inlinetest-disable",
        action="store_true",
        default=False,
        help="disable inlinetests",
        dest="inlinetest_disable",
    )
    parser.addoption(
        "--inlinetest-group",
        action="append",
        default=[],
        metavar="tag",
        help="group inlinetests",
        dest="inlinetest_group",
    )
    parser.addoption(
        "--inlinetest-order",
        action="append",
        default=[],
        metavar="tag",
        help="order inlinetests",
        dest="inlinetest_order",
    )


@pytest.hookimpl()
def pytest_exception_interact(node, call, report):
    if isinstance(call.excinfo.value, MalformedException) or isinstance(call.excinfo.value, AssertionError):
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
    if not isinstance(collector, HIGHLEVEL_SCOPES):
        if collector.config.getoption("inlinetest_only") and (not isinstance(collector, InlinetestModule)):
            collector.collect = lambda: []  # type: ignore[assignment]
        if collector.config.getoption("inlinetest_disable") and isinstance(collector, InlinetestModule):
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
    def __init__(self):
        self.assume_stmts = []
        self.check_stmts = []
        self.given_stmts = []
        self.previous_stmts = []
        self.import_stmts = []
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
        self.devices = None
        self.globs = {}

    def write_imports(self):
        import_str = ""
        for n in self.import_stmts:
              import_str += ExtractInlineTest.node_to_source_code(n) + "\n"
        return import_str

    def to_test(self):
        prefix = "\n"
        
        # for n in self.import_stmts:
        #     import_str += ExtractInlineTest.node_to_source_code(n) + "\n"
        
        
        if self.prev_stmt_type == PrevStmtType.CondExpr:
            if self.assume_stmts == []:
                return prefix.join(
                    [ExtractInlineTest.node_to_source_code(n) for n in self.given_stmts]
                    + [ExtractInlineTest.node_to_source_code(n) for n in self.check_stmts]
                )
            else:
                body_nodes = (
                    [n for n in self.given_stmts] + [n for n in self.previous_stmts] + [n for n in self.check_stmts]
                )
                assume_statement = self.assume_stmts[0]
                assume_node = self.build_assume_node(assume_statement, body_nodes)
                return prefix.join(ExtractInlineTest.node_to_source_code(assume_node))

        else:
            if self.assume_stmts is None or self.assume_stmts == []:
                return prefix.join(
                    [ExtractInlineTest.node_to_source_code(n) for n in self.given_stmts]
                    + [ExtractInlineTest.node_to_source_code(n) for n in self.previous_stmts]
                    + [ExtractInlineTest.node_to_source_code(n) for n in self.check_stmts]
                )
            else:
                body_nodes = (
                    [n for n in self.given_stmts] + [n for n in self.previous_stmts] + [n for n in self.check_stmts]
                )
                assume_statement = self.assume_stmts[0]
                assume_node = self.build_assume_node(assume_statement, body_nodes)
                return prefix.join([ExtractInlineTest.node_to_source_code(assume_node)])

    def build_assume_node(self, assumption_node, body_nodes):
        return ast.If(assumption_node, body_nodes, [])

    def __repr__(self):
        if self.test_name:
            return f"inline test {self.test_name}, starting at line {self.lineno}"
        else:
            return f"inline test, starting at line {self.lineno}"

    def is_empty(self) -> bool:
        return not self.check_stmts

    def __eq__(self, other):
        return (
            self.assume_stmts == other.assume_stmts
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
                    node.children = [child] if not hasattr(node, "children") else [child] + node.children

        extract_inline_test = ExtractInlineTest()
        extract_inline_test.visit(tree)
        if globs:
            for inline_test in extract_inline_test.inline_test_list:
                inline_test.globs = copy.copy(globs)
        return extract_inline_test.inline_test_list


class ExtractInlineTest(ast.NodeTransformer):
    package_name_str = "inline"
    class_name_str = "itest"
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
    diff_given_str = "diff_given"
    group_str = "Group"
    arg_test_name_str = "test_name"
    arg_parameterized_str = "parameterized"
    arg_repeated_str = "repeated"
    arg_tag_str = "tag"
    arg_disabled_str = "disabled"
    arg_timeout_str = "timeout"
    arg_devices_str = "devices"
    diff_test_str = "diff_test"
    assume = "assume"
    
    import_str = "import"
    from_str = "from"
    as_str = "as"
    
    inline_module_imported = False

    def __init__(self):
        self.cur_inline_test = InlineTest()
        self.inline_test_list = []
        self.import_list = []

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
        if node.module == self.package_name_str and node.names[0].name == self.class_name_str:
            self.inline_module_imported = True
        return self.generic_visit(node)

    def find_condition_stmt(self, stmt_node):
        if isinstance(stmt_node.parent, ast.If) or isinstance(stmt_node.parent, ast.While):
            self.cur_inline_test.prev_stmt_type = PrevStmtType.CondExpr
            return stmt_node.parent.test
        else:
            raise NotImplementedError("inline test: failed to find a tested statement")

    def find_previous_stmt(self, node):
        # get the previous stmt that is not itest() by finding the previous sibling
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
                if isinstance(prev_stmt_node.value, ast.Call) and self.is_inline_test_class(prev_stmt_node.value):
                    continue
                else:
                    return prev_stmt_node
            return self.find_condition_stmt(prev_stmt_node)

    def collect_inline_test_calls(self, node, inline_test_calls: List[ast.Call], import_calls: List[ast.Import], import_from_calls: List[ast.ImportFrom]):
        """
        collect all function calls in the node
        """
        if isinstance(node, ast.Attribute):
            self.collect_inline_test_calls(node.value, inline_test_calls, import_calls, import_from_calls)
        elif isinstance(node, ast.Call):
            inline_test_calls.append(node)
            self.collect_inline_test_calls(node.func, inline_test_calls, import_calls, import_from_calls)
        elif isinstance(node, ast.Import):
            import_calls.append(node)
            self.collect_inline_test_calls(node.func, inline_test_calls, import_calls, import_from_calls)
        elif isinstance(node, ast.ImportFrom):
            import_from_calls.append(node)
            self.collect_inline_test_calls(node.func, inline_test_calls, import_calls, import_from_calls)

    def collect_import_calls(self, node, import_calls: List[ast.Import], import_from_calls: List[ast.ImportFrom]):
        """
        collect all import calls in the node (should be done first)
        """

        while not isinstance(node, ast.Module) and node.parent != None:
            node = node.parent
       
        if not isinstance(node, ast.Module):
            return
            
        for child in node.children:
            if isinstance(child, ast.Import):
                import_calls.append(child)
            elif isinstance(child, ast.ImportFrom):
                import_from_calls.append(child)

    def parse_constructor(self, node):
        """
        Parse a constructor call.
        """
        
        # Argument Order:
        # 0) test_name (str)
        # 1) parameterized (bool)
        # 2) repeated (positive integer)
        # 3) tag (str)
        # 4) disabled (bool)
        # 5) timeout (positive float)
        # 6) devices (str array)
        
        
        
        keyword_idxs = {
            self.arg_test_name_str : 0,
            self.arg_parameterized_str : 1,
            self.arg_repeated_str : 2,
            self.arg_tag_str : 3,
            self.arg_disabled_str : 4,
            self.arg_timeout_str : 5,
            self.arg_devices_str : 6
        }
        
        NUM_OF_ARGUMENTS = 7
        if len(node.args) + len(node.keywords) <= NUM_OF_ARGUMENTS:
            # positional arguments
            self.parse_constructor_args(node.args)
            
            #keyword arguments
            keyword_args = []
            
            #create list with 7 null values (for each position)
            for i in range(0, NUM_OF_ARGUMENTS):
                keyword_args.append(None)
           
            for keyword in node.keywords:
                keyword_args[keyword_idxs[keyword.arg].value] = keyword.value
            self.parse_constructor_args(keyword_args)


        if not self.cur_inline_test.test_name:
            # by default, use lineno as test name
            self.cur_inline_test.test_name = f"line{node.lineno}"
        # set the line number
        self.cur_inline_test.lineno = node.lineno

    def parse_constructor_args(self, args):
        class ConstrArgs(enum.Enum):
            TEST_NAME = 0
            PARAMETERIZED = 1
            REPEATED = 2
            TAG_STR = 3
            DISABLED = 4
            TIMEOUT = 5
            DEVICES = 6
        
        property_names = {
            ConstrArgs.TEST_NAME : "test_name",
            ConstrArgs.PARAMETERIZED : "parameterized",
            ConstrArgs.REPEATED : "repeated",
            ConstrArgs.TAG_STR : "tag",
            ConstrArgs.DISABLED : "disabled",
            ConstrArgs.TIMEOUT : "timeout",
            ConstrArgs.DEVICES : "devices"
        }
            
        pre_38_val_names = {
            ConstrArgs.TEST_NAME : "s",
            ConstrArgs.PARAMETERIZED : "value",
            ConstrArgs.REPEATED : "n",
            ConstrArgs.TAG_STR : "s",
            ConstrArgs.DISABLED : "value",
            ConstrArgs.TIMEOUT : "n",
            ConstrArgs.DEVICES : ""
        }
                
        pre_38_expec_ast_arg_type = {
            ConstrArgs.TEST_NAME : ast.Str,
            ConstrArgs.PARAMETERIZED : ast.NameConstant,
            ConstrArgs.REPEATED : ast.Num,
            ConstrArgs.TAG_STR : ast.List,
            ConstrArgs.DISABLED : ast.NameConstant,
            ConstrArgs.TIMEOUT : ast.Num,
        }
        
        expected_ast_arg_type = { 
            ConstrArgs.TEST_NAME : ast.Constant,
            ConstrArgs.PARAMETERIZED : ast.Constant,
            ConstrArgs.REPEATED : ast.Constant,
            ConstrArgs.TAG_STR : ast.List,
            ConstrArgs.DISABLED : ast.Constant,
            ConstrArgs.TIMEOUT : ast.Constant
        }
        
        expected_ast_val_args = {
            ConstrArgs.TEST_NAME : [str],
            ConstrArgs.PARAMETERIZED : [bool],
            ConstrArgs.REPEATED : [int],
            ConstrArgs.TAG_STR : [None],
            ConstrArgs.DISABLED : [bool],
            ConstrArgs.TIMEOUT : [float, int],
            ConstrArgs.DEVICES : [str]
        }
        
        NUM_OF_ARGUMENTS = 7
        
        # Arguments organized by expected ast type, value type, and index in that order
        for index, arg in enumerate(args):
            # Skips over null arguments; needed for keywords
            if arg == None:
                continue
            
            # Devices are not referenced in versions before 3.8; all other arguments can be from any version
            if index == ConstrArgs.DEVICES and isinstance(arg, ast.List):
                devices = []
                for elt in arg.elts:
                    if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                        raise MalformedException("devices can only be List of string")
                    if elt.value not in {"cpu", "cuda", "mps"}:
                        raise MalformedException(f"Invalid device: {elt.value}. Must be one of ['cpu', 'cuda', 'mps']")
                    devices.append(elt.value)
                self.cur_inline_test.devices = devices
            # Assumes version is past 3.8, no explicit references to ast.Constant before 3.8
            else:
                corr_arg_type = False
                corr_val_type = False
                value_prop_name = ""
                arg_idx = ConstrArgs(index)
                
                if sys.version_info >= (3, 8, 0) and isinstance(arg, expected_ast_arg_type[arg_idx]):
                    corr_arg_type = True
                    value_prop_name = "value"
                elif sys.version_info < (3, 8, 0) and isinstance(arg, pre_38_expec_ast_arg_type[arg_idx]):
                    corr_arg_type = True
                    value_prop_name = pre_38_val_names[arg_idx]
                
                # Verifies value types; skipped for ast node types with no nested values
                for arg_type in expected_ast_val_args[arg_idx]:
                    if arg_type == None:
                        corr_val_type = True
                        break
                    if isinstance(arg.value, arg_type):
                        corr_val_type = True
                        break
                
                if corr_val_type and corr_arg_type:
                    # Accounts for additional checks for REPEATED and TAG_STR arguments
                    if arg_idx == ConstrArgs.REPEATED:
                        if arg.value <= 0:
                            raise MalformedException(f"inline test: {self.arg_repeated_str} must be greater than 0")
                        self.cur_inline_test.repeated = getattr(arg, value_prop_name)
                    elif arg_idx == ConstrArgs.TAG_STR:
                        tags = []
                        for elt in arg.elts:
                            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                                raise MalformedException(f"tag can only be List of string")
                            tags.append(getattr(elt, value_prop_name))
                        self.cur_inline_test.tag = tags
                    # For non-special cases, set the attribute defined by the dictionary
                    else:
                        setattr(self.cur_inline_test,
                                property_names[arg_idx],
                                getattr(arg, value_prop_name))
                    
                   
                    
                    # match arg_idx:
                    #     case ConstrArgs.REPEATED:
                    #         if arg.value <= 0:
                    #             raise MalformedException(f"inline test: {self.arg_repeated_str} must be greater than 0")
                    #         self.cur_inline_test.repeated = getattr(arg, value_prop_name)
                    #     case ConstrArgs.TAG_STR:
                    #         tags = []
                    #         for elt in arg.elts:
                    #             if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                    #                 raise MalformedException(f"tag can only be List of string")
                    #             tags.append(getattr(elt, value_prop_name))
                    #         self.cur_inline_test.tag = tags
                    #     # For non-special cases, set the attribute defined by the dictionary
                    #     case _:
                    #         setattr(self.cur_inline_test,
                    #                 property_names[arg_idx],
                    #                 getattr(arg, value_prop_name))
                else:
                    raise MalformedException(
                        f"inline test: {self.class_name_str}() accepts {NUM_OF_ARGUMENTS} arguments. 'test_name' must be a string constant, 'parameterized' must be a boolean constant, 'repeated' must be a positive integer, 'tag' must be a list of string, 'timeout' must be a positive float"
                    )
                    #raise MalformedException("Argument " + str(index) + " incorrectly formatted. Argument should be a " + ConstrArgs.expected_ast_val_args[index].type())

    def parameterized_inline_tests_init(self, node: ast.List):
        if not self.cur_inline_test.parameterized_inline_tests:
            self.cur_inline_test.parameterized_inline_tests = [InlineTest() for _ in range(len(node.elts))]
        if len(node.elts) != len(self.cur_inline_test.parameterized_inline_tests):
            raise MalformedException("inline test: parameterized tests must have the same number of test cases")

    def parse_given(self, node):
        if len(node.args) == 2:
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[1])
                for index, value in enumerate(node.args[1].elts):
                    assign_node = ast.Assign(targets=[node.args[0]], value=value)
                    self.cur_inline_test.parameterized_inline_tests[index].given_stmts.append(assign_node)
            else:
                assign_node = ast.Assign(targets=[node.args[0]], value=node.args[1])
                self.cur_inline_test.given_stmts.append(assign_node)
        else:
            raise MalformedException("inline test: invalid given(), expected 2 args")

    def parse_diff_given(self, node):
        PROPERTY = 0
        VALUES = 1
        
        if len(node.args) == 2:
            if self.cur_inline_test.parameterized:
                raise MalformedException("inline test: diff_given() does not currently support parameterized inline tests.")
            else:
                devices = []
                for elt in node.args[VALUES].elts:
                    if elt.value not in {"cpu", "cuda", "mps"}:
                        raise MalformedException(f"Invalid device: {elt.value}. Must be one of ['cpu', 'cuda', 'mps']")
                    devices.append(elt.value)
                setattr(self.cur_inline_test, node.args[PROPERTY].id, devices)
        else:
            raise MalformedException("inline test: invalid diff_given(), expected 2 args")

    def parse_assume(self, node):
        if len(node.args) == 1:
            if self.cur_inline_test.parameterized:
                self.parameterized_inline_tests_init(node.args[0])
                for index, value in enumerate(node.args[0].elts):
                    test_node = self.parse_group(value)
                    assumption_node = self.build_assume(test_node)
                    self.cur_inline_test.parameterized_inline_tests[index].assume_stmts.append(assumption_node)
            else:
                test_node = self.parse_group(node.args[0])
                self.cur_inline_test.assume_stmts.append(test_node)
        else:
            raise MalformedException("inline test: invalid assume() call, expected 1 arg")

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
                    self.cur_inline_test.parameterized_inline_tests[index].check_stmts.append(assert_node)
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
                    ast.Constant("bool({0}) is True\nActual: bool({1}) is False\nExpected: bool({1}) is True\n"),
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
                    self.cur_inline_test.parameterized_inline_tests[index].check_stmts.append(assert_node)
            else:
                test_node = self.parse_group(node.args[0])
                assert_node = self.build_assert_true(test_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException("inline test: invalid check_true(), expected 1 arg")

    def build_assert_false(self, operand_node):
        assert_node = ast.Assert(
            test=ast.UnaryOp(op=ast.Not(), operand=operand_node),
            msg=ast.Call(
                func=ast.Attribute(
                    ast.Constant("bool({0}) is False\nActual: bool({1}) is True\nExpected: bool({1}) is False\n"),
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
                    self.cur_inline_test.parameterized_inline_tests[index].check_stmts.append(assert_node)
            else:
                operand_node = self.parse_group(node.args[0])
                assert_node = self.build_assert_false(operand_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException("inline test: invalid check_false(), expected 1 arg")

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
                    self.cur_inline_test.parameterized_inline_tests[index].check_stmts.append(assert_node)
            else:
                comparator_node = self.parse_group(node.args[1])
                assert_node = self.build_assert_neq(left_node, comparator_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException("inline test: invalid check_neq(), expected 2 args")

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
                    self.cur_inline_test.parameterized_inline_tests[index].check_stmts.append(assert_node)
            else:
                operand_node = self.parse_group(node.args[0])
                assert_node = self.build_assert_none(operand_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException("inline test: invalid check_none(), expected 1 arg")

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
                    self.cur_inline_test.parameterized_inline_tests[index].check_stmts.append(assert_node)
            else:
                operand_node = self.parse_group(node.args[0])
                assert_node = self.build_assert_not_none(operand_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException("inline test: invalid check_not_none(), expected 1 arg")

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
                    self.cur_inline_test.parameterized_inline_tests[index].check_stmts.append(assert_node)
            else:
                comparator_node = self.parse_group(node.args[1])
                assert_node = self.build_assert_same(left_node, comparator_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException("inline test: invalid check_same(), expected 2 args")

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
                    self.cur_inline_test.parameterized_inline_tests[index].check_stmts.append(assert_node)
            else:
                comparator_node = self.parse_group(node.args[1])
                assert_node = self.build_assert_not_same(left_node, comparator_node)
                self.cur_inline_test.check_stmts.append(assert_node)
        else:
            raise MalformedException("inline test: invalid check_not_same(), expected 2 args")
    
    def parse_diff_test(self, node):
        if not self.cur_inline_test.devices:
            raise MalformedException("diff_test can only be used with the 'devices' parameter.")

        if len(node.args) != 1:
            raise MalformedException("diff_test() requires exactly 1 argument.")

        output_node = self.parse_group(node.args[0])
        
        # Get the original operation
        original_op = None
        for stmt in self.cur_inline_test.previous_stmts:
            if isinstance(stmt, ast.Assign) and stmt.targets[0].id == output_node.id:
                original_op = stmt.value
                break
        
        if not original_op:
            raise MalformedException("Could not find original operation for diff_test")
        
        # Create our new statements
        new_statements = []
        device_outputs = []
        
        # Import necessary modules for seed setting - Always add these
        # Import random module
        import_random = ast.ImportFrom(
            module='random',
            names=[ast.alias(name='seed', asname=None)],
            level=0
        )
        new_statements.append(import_random)
        
        # Import numpy.random
        import_np = ast.ImportFrom(
            module='numpy',
            names=[ast.alias(name='random', asname='np_random')],
            level=0
        )
        new_statements.append(import_np)
        
        # Create seed function - Always add this
        seed_func_def = ast.FunctionDef(
            name='set_random_seed',
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg='seed_value', annotation=None)],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id='seed', ctx=ast.Load()),
                        args=[ast.Name(id='seed_value', ctx=ast.Load())],
                        keywords=[]
                    )
                ),
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='torch', ctx=ast.Load()),
                            attr='manual_seed'
                        ),
                        args=[ast.Name(id='seed_value', ctx=ast.Load())],
                        keywords=[]
                    )
                ),
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='np_random', ctx=ast.Load()),
                            attr='seed'
                        ),
                        args=[ast.Name(id='seed_value', ctx=ast.Load())],
                        keywords=[]
                    )
                )
            ],
            decorator_list=[],
            returns=None
        )
        new_statements.append(seed_func_def)

        # Process input tensors
        for given_stmt in self.cur_inline_test.given_stmts:
            input_var = given_stmt.targets[0].id
            ref_var = f"{input_var}_ref"
            
            # Always clone inputs for in-place operations
            new_statements.append(
                ast.Assign(
                    targets=[ast.Name(id=ref_var, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=given_stmt.value,
                            attr="clone"
                        ),
                        args=[],
                        keywords=[]
                    )
                )
            )
            
            # Create device-specific versions
            for device in self.cur_inline_test.devices:
                device_var = f"{input_var}_{device}"
                
                new_statements.append(
                    ast.Assign(
                        targets=[ast.Name(id=device_var, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=ref_var, ctx=ast.Load()),
                                attr="to"
                            ),
                            args=[ast.Constant(value=device)],
                            keywords=[]
                        )
                    )
                )
        
        # Create device-specific operations
        device_input_map = {device: {} for device in self.cur_inline_test.devices}
        for device in self.cur_inline_test.devices:
            for given_stmt in self.cur_inline_test.given_stmts:
                input_var = given_stmt.targets[0].id
                device_input_map[device][input_var] = f"{input_var}_{device}"
            
            # Always set seed before each device operation - no condition check
            new_statements.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id='set_random_seed', ctx=ast.Load()),
                        args=[ast.Constant(value=42)],  # Use constant seed 42
                        keywords=[]
                    )
                )
            )
                
            device_op = copy.deepcopy(original_op)
            
            # Replace input references
            class ReplaceInputs(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id in device_input_map[device]:
                        return ast.Name(id=device_input_map[device][node.id], ctx=node.ctx)
                    return node
            
            device_op = ReplaceInputs().visit(device_op)
            device_output = f"output_{device}"
            
            new_statements.append(
                ast.Assign(
                    targets=[ast.Name(id=device_output, ctx=ast.Store())],
                    value=device_op
                )
            )
            device_outputs.append(device_output)
        
        # Standard comparison method for all operations - no condition check
        comparisons = []
        for i in range(len(device_outputs) - 1):
            dev1 = device_outputs[i]
            dev2 = device_outputs[i + 1]
            
            dev1_cpu = f"{dev1}_cpu"
            dev2_cpu = f"{dev2}_cpu"
            
            # Move outputs back to CPU for comparison
            new_statements.append(
                ast.Assign(
                    targets=[ast.Name(id=dev1_cpu, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=dev1, ctx=ast.Load()),
                            attr="to"
                        ),
                        args=[ast.Constant(value="cpu")],
                        keywords=[]
                    )
                )
            )
            
            new_statements.append(
                ast.Assign(
                    targets=[ast.Name(id=dev2_cpu, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=dev2, ctx=ast.Load()),
                            attr="to"
                        ),
                        args=[ast.Constant(value="cpu")],
                        keywords=[]
                    )
                )
            )
            
            # Standard allclose comparison
            comparison = self.build_assert_eq(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=dev1_cpu, ctx=ast.Load()),
                        attr="allclose"
                    ),
                    args=[
                        ast.Name(id=dev2_cpu, ctx=ast.Load())
                    ],
                    keywords=[
                        ast.keyword(arg="rtol", value=ast.Constant(value=1e-4)),
                        ast.keyword(arg="atol", value=ast.Constant(value=1e-4)),
                        ast.keyword(arg="equal_nan", value=ast.Constant(value=True))
                    ]
                ),
                ast.Constant(value=True)
            )
            comparisons.append(comparison)
        
        # Replace statements
        self.cur_inline_test.previous_stmts = new_statements
        self.cur_inline_test.check_stmts = comparisons
        
    def parse_import(self, node):
        # TODO: Differentiate between import, from import, and import alias
        import_node = ast.Import(
            names=[
                ast.alias(name=node)
            ]
        )
        return import_node
    
    def parse_import_from(self, node):
        pass

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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == self.group_str:
            # node type is ast.Call, node.func type is ast.Name
            if sys.version_info >= (3, 8, 0):
                index_args = [arg.value for arg in node.args]
            else:
                # python3.7 type of arg is ast.Num which does not in higher version
                index_args = [arg.n for arg in node.args]
            if self.cur_inline_test.prev_stmt_type != PrevStmtType.CondExpr:
                raise MalformedException("inline test: Group() must be called to test a conditional statement")
            if not self.cur_inline_test.previous_stmts:
                raise MalformedException("inline test: previous statement not found")
            stmt = self.cur_inline_test.previous_stmts[0]
            for i, index_arg in enumerate(index_args):
                if isinstance(stmt, ast.BoolOp):
                    if index_arg < 0 or index_arg >= len(stmt.values):
                        raise MalformedException(f"inline test: Group() {i} index with value {index_arg} out of range")
                    else:
                        stmt = stmt.values[index_arg]
                        # raise NotImplementedError(index_arg, ast.dump(stmt))
            return stmt
        else:
            return node
            

    def parse_parameterized_test(self):
        for index, parameterized_test in enumerate(self.cur_inline_test.parameterized_inline_tests):
            parameterized_test.previous_stmts = self.cur_inline_test.previous_stmts
            parameterized_test.prev_stmt_type = self.cur_inline_test.prev_stmt_type
            parameterized_test.lineno = self.cur_inline_test.lineno
            parameterized_test.test_name = self.cur_inline_test.test_name + "_" + str(index)

    def parse_inline_test(self, node):
        import_calls = []
        import_from_calls = []
        inline_test_calls = [] 
        
        self.collect_inline_test_calls(node, inline_test_calls, import_calls, import_from_calls)
        self.collect_import_calls(node, import_calls, import_from_calls)
        
        inline_test_calls.reverse()

        if len(inline_test_calls) <= 1:
            raise MalformedException("inline test: invalid inline test, requires at least one assertion")

        # "itest()" or "itest('test name')" or "itest('test name', True)" or "itest(parameterized=True)" or "itest(test_name='test name', parameterized=True)"
        constructor_call = inline_test_calls[0]
        if isinstance(constructor_call.func, ast.Name) and constructor_call.func.id == self.class_name_str:
            self.parse_constructor(constructor_call)
        else:
            raise MalformedException("inline test: invalid inline test constructor")

        # "assume_true(...) or assume_false(...)
        inline_test_call_index = 1
        if len(inline_test_calls) >= 2:
            call = inline_test_calls[1]
            if isinstance(call.func, ast.Attribute) and call.func.attr == self.assume:
                self.parse_assume(call)
                inline_test_call_index += 1

        for call in inline_test_calls[inline_test_call_index:]:
            if isinstance(call.func, ast.Attribute):
                if call.func.attr == self.given_str:
                    self.parse_given(call)
                    inline_test_call_index += 1
                elif call.func.attr == self.diff_given_str:
                    self.parse_diff_given(call)
                    inline_test_call_index += 1
                
                # match call.func.attr:
                #     # "given(a, 1)"
                #     case self.given_str:
                #         self.parse_given(call)
                #         inline_test_call_index += 1
                #      # "diff_given(devices, ["cpu", "cuda"])"
                #     case self.diff_given_str:
                #         self.parse_diff_given(call)
                #         inline_test_call_index += 1
            else:
                break

        for import_stmt in import_calls:
            self.cur_inline_test.import_stmts.append(import_stmt)
        for import_stmt in import_from_calls:
            self.cur_inline_test.import_stmts.append(import_stmt)


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
            elif call.func.attr == self.diff_test_str:
                self.parse_diff_test(call)
            elif call.func.attr == self.fail_str:
                self.parse_fail(call)
            elif call.func.attr == self.given_str:
                raise MalformedException(
                    f"inline test: given() must be called before check_eq()/check_true()/check_false()/diff_test()"
                )
            else:
                raise MalformedException(f"inline test: invalid function call {self.node_to_source_code(call.func)}")

        if self.cur_inline_test.parameterized:
            self.parse_parameterized_test()
            self.inline_test_list.extend(self.cur_inline_test.parameterized_inline_tests)
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
    # Finder should NOT store any global variables
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

    # def find_imports(self, obj, module=None):
    #     if module is False:
    #         module = None
    #     elif module is None:
    #         module = inspect.getmodule(obj)
        

    def find(self, obj, module=None, globs=None, extraglobs=None, imports=None):
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

        # Find intersection between loaded modules and module imports
        # if imports is None:
        #     imports = set(sys.modules) & set(globs)
        # else:
        #     imports = imports.copy()

        # Recursively explore `obj`, extracting InlineTests.
        tests = []
        self._find(tests, obj, module, globs, imports, {})
        return tests

    def _find(self, tests, obj, module, globs, imports, seen):
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
                if (self._is_routine(val) or inspect.isclass(val)) and self._from_module(module, val):
                    self._find(tests, val, module, globs, imports, seen)

        # Look for tests in a class's contained objects.
        if inspect.isclass(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                # Special handling for staticmethod/classmethod.
                if isinstance(val, (staticmethod, classmethod)):
                    val = val.__func__

                # Recurse to methods, properties, and nested classes.
                if (inspect.isroutine(val) or inspect.isclass(val) or isinstance(val, property)) and self._from_module(
                    module, val
                ):
                    valname = "%s" % (valname)
                    self._find(tests, val, module, globs, imports, seen)


######################################################################
## InlineTest Runner
######################################################################
class InlineTestRunner:
    def run(self, test: InlineTest, out: List) -> None:
        test_str = test.write_imports()
        test_str += test.to_test()
        print(test_str)
        tree = ast.parse(test_str)
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
        self._init_fixtureinfo_request()

    # fixture API changed in pytest 8
    # https://github.com/pytest-dev/pytest/issues/11218
    if pytest.version_tuple >= (8, 0, 0):

        def _init_fixtureinfo_request(self) -> None:
            self.funcargs: Dict[str, object] = {}
            fm = self.session._fixturemanager
            fixtureinfo = fm.getfixtureinfo(node=self, func=None, cls=None)
            self._fixtureinfo = fixtureinfo
            self.fixturenames = fixtureinfo.names_closure
            self._request = TopRequest(self, _ispytest=True)  # type: ignore[arg-type]

    else:

        def _init_fixtureinfo_request(self) -> None:
            def func() -> None:
                pass

            self.funcargs: Dict[str, object] = {}
            fm = self.session._fixturemanager
            self._fixtureinfo = fm.getfixtureinfo(  # type: ignore[attr-defined]
                node=self, func=func, cls=None, funcargs=False
            )
            self._request = FixtureRequest(self, _ispytest=True)  # type: ignore[arg-type]

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
            self._request._fillfixtures()
            globs = dict(getfixture=self._request.getfixturevalue)
            for name, value in self._request.getfixturevalue("inlinetest_namespace").items():
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
        prio_sorted = [val for (_, val) in sorted(zip(sorted_ordering, prio_unsorted), key=lambda x: x[0])]
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
            except Exception as e:
                # (ImportError, ModuleNotFoundError, TypeError, NameError, FileNotFoundError)
                if self.config.getvalue("inlinetest_ignore_import_errors"):
                    pytest.skip("unable to import module %r" % self.path)
                else:
                    raise

        finder = InlineTestFinder()
        runner = InlineTestRunner()

        group_tags = self.config.getoption("inlinetest_group", default=None)
        order_tags = self.config.getoption("inlinetest_order", default=None)

        # TODO: import all modules through the finder first before extracting inline tests
        # - Create ast for all imports
        # - If a function references an import, then include the imported library reference in the ast

        for test_list in finder.find(module):
            # reorder the list if there are tests to be ordered
            ordered_list = InlinetestModule.order_tests(test_list, order_tags)
            if ordered_list is not None:
                for test in ordered_list:
                    if (
                        test.is_empty() or (group_tags and len(set(test.tag) & set(group_tags)) == 0) or test.disabled
                    ):  # skip empty inline tests and tests with tags not in the tag list and disabled tests
                        continue

                    yield InlinetestItem.from_parent(
                        self,
                        name=test.test_name,
                        runner=runner,
                        dtest=test,
                    )


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
