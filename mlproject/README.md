### In juwelsbooster
1. Set python to version 3.9. For this load the following modules:
```
ml --force purge
ml use $OTHERSTAGES
ml Stages/2022
ml GCCcore/.11.2.0
ml Python/3.9.6
```

2. Create a virtual enviroment and activate it:
```
python -m venv <venv-name>
source <venv-name>/bin/activate
```

3. Clone this Git repository and checkout the `main` branch

```
git clone https://github.com/saragrau4/maelstrom-radiation.git
git checkout main
```

3. Install ap3 dependencies with pip. 
```
pip install -r requirements_wo_modules.txt
```

4. Add the following code at the end of the virtual enviroment activate file `<venv-name>/bin/activate`:
```
BASE_DIR="<absolute path to maelstrom-radiation/climetlab_maelstrom_radiation/benchmarks directory>"
# expand PYTHONPATH
export PYTHONPATH=${BASE_DIR}:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/utils:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/handle_data:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/models:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/postprocess:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/preprocess:$PYTHONPATH

```
### In mantik

Set up a project in Mantik to enable the execution of your experiment. For a step-by-step guide, refer to the quickstart tutorial available [here](https://mantik-ai.gitlab.io/mantik/ui/quickstart.html)

### In your local mlproject

1. Set `Python` in `unicore-config-venv.yaml` to the path of your virtual enviroment

```
Environment:
  Python: /path/to/<venv-name>
```

2. Run your experiment with mantik
```
mantik runs submit <absolute path to maelstrom-radiation/mlproject directory> --backend-config unicore-config-venv.yaml --entry-point main --experiment-id <experiment ID> -v
```