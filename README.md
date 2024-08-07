# Installation

## Conda
1. Create Conda environment with python=3.10: `conda create -n elk-experiments python=3.10` 
2. Disable poetry virtualenvs `poetry config virtualenvs.create false`
3. Run poetry install `poetry install`

# Converting Notebooks to Scripts

```
jupyter nbconvert --to python  notebooks/hypothesis_tests.ipynb --output-dir  scripts/
```