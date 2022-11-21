#!/bin/bash

_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [[ $OSTYPE == 'darwin'* ]]; then
        MINICONDA_URL='https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh'
else
        MINICONDA_URL='https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
fi


function get_or_install_conda_path() {
        local conda_exe=$(which conda)
        if [[ ! -z ${conda_exe} ]]; then
                echo "Found existing conda executable at $(which conda)" 1>&2
                echo "$(dirname ${conda_exe})/../etc/profile.d/conda.sh"
                return
        fi

        local install_dir=${_DIR}/miniconda
        rm -rf ${install_dir}

        local temp_dir=$(mktemp -d)
        ( cd ${temp_dir}
          wget $MINICONDA_URL -O miniconda.sh 1>&2
          chmod +x miniconda.sh
          ./miniconda.sh -b -p ${install_dir} 1>&2
          if [[ $? -ne 0 ]]; then
                  echo "Failed to install miniconda! Please install miniconda following the instructions at this link, then rerun the script: https://docs.conda.io/en/latest/miniconda.html" 1>&2
                  exit 1
          fi
        )

        rm -rf ${temp_dir}

        echo "Installation of miniconda successful" 1>&2

        echo "${install_dir}/etc/profile.d/conda.sh"
}


function prepare_conda_env() {
        ### Preparing the base environment "inline-dev"
        local env_name="inline-dev"
        local conda_path=$(get_or_install_conda_path)

        echo ">>> Preparing conda environment \"${env_name}\", using conda at ${conda_path}" 1>&2
        
        # Preparation
        source ${conda_path}
        set -e
        conda env remove --name $env_name
        conda create --name $env_name python=3.9 pip -y
        conda activate $env_name

        pip install .[dev]
}

prepare_conda_env
conda activate inline-dev
