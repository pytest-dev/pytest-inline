#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Script needs 1 arguments: python file dir"
    echo "Example: bash parallel_tests.sh ./test_files"
else
    script_dir=$( cd $( dirname $0 ) && pwd )
    logs_dir="$script_dir/logs"

    mkdir -p $logs_dir

    pip uninstall -y pytest-xdist

    echo "Running non-parallel..."
    ( time pytest $1 ) >> "$logs_dir/non-parallel"
    echo "Finished non-parallel"

    pip install pytest-xdist

    echo "Running parallel..."
    ( time pytest -n auto $1 ) >> "$logs_dir/parallel"
    echo "Finished parallel"

    echo "Logs in: $logs_dir"
fi
