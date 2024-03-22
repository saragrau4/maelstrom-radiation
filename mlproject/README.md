# Running Application 3 with Mantik CLI
To run this application in juwels-booster with the mantik CLI follow these instructions:

1. Login to juwels-booster via SSH. To access juwels-booster via SSH, please follow the instructions provided in this [tutorial](https://apps.fz-juelich.de/jsc/hps/juwels/access.html#ssh-login)

2. Once you are logged in on one of JSC or CSCS compute resources, set python to version 3.9. For this load the following modules:

```
# JSC
ml --force purge
ml use $OTHERSTAGES
ml Stages/2022
ml GCCcore/.11.2.0
ml Python/3.9.6
```

```
# CSCS
module load cray-python/3.9.4.1 cudatoolkit/21.5_11.3
```

3. Create a virtual enviroment and activate it:
```
python -m venv <venv-name>
source <venv-name>/bin/activate
```

4. Clone this Git repository and checkout the `main` branch

```
git clone https://github.com/saragrau4/maelstrom-radiation.git
git checkout main
```

5. Install ap3 dependencies with pip. 
```
pip install -r requirements_wo_modules.txt
```

6. Add the following code at the end of the virtual enviroment activation file `<venv-name>/bin/activate`:
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

7. The results will be logged to an Experiment on the MLflow tracking server on Mantik. Set up a project in Mantik and create a new Experiment. Note its experiment Id, which will be needed in the submission command. For a step-by-step guide, refer to the Quickstart tutorial available [here](https://mantik-ai.gitlab.io/mantik/ui/quickstart.html).

8. Update the `unicore-config-venv.yaml` or `firecrest-config-venv.yaml` file according to the HPC site where you intend to submit your run. Specify the `PreRunCommandOnCompute` parameter with the path to your virtual environment.

<pre><code># JSC
PreRunCommandOnComputeNode:
    Command: > 
      module load Stages/2022 GCCcore/.11.2.0 GCC/11.2.0 cuDNN/8.3.1.22-CUDA-11.5 Python/3.9.6
      source <b>/path/to/&lt;venv-name&gt;</b>/bin/activate;
</code></pre>


<pre><code># CSCS
PreRunCommandOnComputeNode:
    Command: > 
      module load cray-python/3.9.4.1 cudatoolkit/21.5_11.3
      source <b>/path/to/&lt;venv-name&gt;</b>/bin/activate;
</code></pre>

9. Run your experiment with mantik
```
mantik runs submit <absolute path to maelstrom-radiation/mlproject directory> --backend-config unicore-config-venv.yaml --entry-point main --experiment-id <experiment ID> --runb-name <run name> -v
```
