conda create -y -n ds_py39 -c conda-forge --override-channels python=3.9 "numpy>=1.16.1" "pandas>=1.1.0" "geopandas>=0.10.2" "psutil>=4.1" "pyarrow>=2.0" "numba>0.51.2" "pyyaml>=5.1" "requests>=2.7" pytest pytest-cov coveralls pycodestyle pytest-regressions jupyter jupyterlab matplotlib descartes pandasql scipy seaborn pyodbc sqlalchemy openpyxl xlrd xlsxwriter sympy nose scikit-learn scikit-learn-intelex autopep8 pip ipykernel
conda env export -n ds_py39 -f environments/ds_py39.yml --no-builds
conda deactivate
