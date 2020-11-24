# LexSubGen
Lexical Substitution Framework

## 1. Installation
Clone LexSubGen repository from github.com.
```shell script
git clone https://github.com/Samsung/LexSubGen
cd LexSubGen
```

### 1.1. Setup anaconda environment
1. Download and install [conda](https://conda.io/docs/user-guide/install/download.html)
2. Create new conda environment
    ```shell script
    conda create -n lexsubgen python=3.7.4
    ```
3. Activate conda environment
    ```shell script
    conda activate lexsubgen
    ```
4. Install requirements
    ```shell script
    pip install -r requirements.txt
    ```
5. Download spacy resources and install context2vec and word_forms from github repositories
    ```shell script
    ./init.sh
    ```
If you want to run nPIC model you should download [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and add
`CORENLP_HOME=/path/to/corenlp/directory` to environment variables. Also
you should add models path to CLASS_PATH environment variable:
```shell script
export CLASSPATH="$CLASSPATH:/path/to/corenlp/javanlp-core.jar:/path/to/corenlp/stanford-corenlp-models-current.jar";
for file in `find /path/to/corenlp -name "*.jar"`; 
do 
    export CLASSPATH="$CLASSPATH:`realpath $file`"; 
done
```

### 1.2. Setup Web Application
**If you do not plan to use the Web Application, skip this section and [go to the next](#13-install-lexsubgen-library)!**
1. Download and install [NodeJS and npm](https://www.npmjs.com/get-npm).
2. Run script for install dependencies and create build files.
```shell script
bash web_app_setup.sh
```

### 1.3. Install lexsubgen library
```shell script
python setup.py install
```

## 3. How to run
### 3.1. Lexical Substitution Evaluation
You can use command line interface to run evaluation.

```
lexsubgen evaluate --config-path CONFIG_PATH 
                   --run-dir RUN_PATH 
                   [--force] 
                   [--additional-modules MODULE_PATH] 
                   [--config CONFIG]
```
**Example:**
```
lexsubgen evaluate --config-path configs/evaluations/lexsub/semeval_all_xlnet.jsonnet
                   --run-dir /runs/test_run 
                   --force
```
##### Arguments:
|Argument        |Default |Description                                                   |
|----------------|--------|--------------------------------------------------------------|
|`--help`        |        |Show this help message and exit                               |
|`--config-path` |`None`  |Path to the configuration file                                |
|`--additional-modules`  |`None`|Path to the additional modules that should be registered|
|`--config`      |`None`  |Configuration                                                 |
|`--force`       |`None`  |Whether to override existing run directory                    |

## Results
Results of the lexical substitution task are presented in the following table. To reproduce them, follow the instructions above to install the correct dependencies. 

<table>
    <thead>
        <tr>
            <th rowspan=2><b>Model</b></th>
            <th colspan=4><b>SemEval</b></th>
            <th colspan=4><b>COINCO</b></th>
        </tr>
        <tr>
            <th>GAP</th>
            <th>P@1</th>
            <th>P@3</th>
            <th>R@10</th>
            <th>GAP</th>
            <th>P@1</th>
            <th>P@3</th>
            <th>R@10</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>XLNet</td>
            <td>59.11</td>
            <td>31.7</td>
            <td>22.8</td>
            <td>34.93</td>
            <td>53.39</td>
            <td>38.16</td>
            <td>28.58</td>
            <td>26.47</td>
        </tr>
        <tr>
            <td>XLNet+embs</td>
            <td><b>59.62</b></td>
            <td><b>49.53</b></td>
            <td><b>34.88</b></td>
            <td><b>47.47</b></td>
            <td><b>55.63</b></td>
            <td><b>51.5</b> </td>
            <td><b>39.92</b></td>
            <td><b>35.12</b></td>
        </tr>
    </tbody>
</table>


### 3.2. Results reproduction
Here we list reproduction commands that correspond
to the results presented in the table above. Besides saving to the 'run-directory' 
all results are saved using mlflow. To check them you can run ```mlflow ui``` in LexSubGen 
directory and then open the web page in a browser. 

Also you can use pytest to check the reproducibility. But it may take a long time:
```shell script
pytest tests/results_reproduction
```
* #### XLNet:
XLNet Semeval07: 
```shell script
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/semeval_xlnet.jsonnet --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet --run-dir='debug/lexsub-all-models/semeval_all_xlnet' --force --experiment-name='lexsub-all-models' --run-name='semeval_all_xlnet'
```

XLNet CoInCo: 
```shell script
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/coinco_xlnet.jsonnet --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet --run-dir='debug/lexsub-all-models/coinco_xlnet' --force --experiment-name='lexsub-all-models' --run-name='coinco_xlnet'
```

XLNet with embeddings similarity Semeval07:
```shell script
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/semeval_xlnet_embs.jsonnet --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet --run-dir='debug/lexsub-all-models/semeval_all_xlnet_embs' --force --experiment-name='lexsub-all-models' --run-name='semeval_all_xlnet_embs'
```

XLNet with embeddings similarity CoInCo:
```shell script
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/coinco_xlnet_embs.jsonnet --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet --run-dir='debug/lexsub-all-models/coinco_xlnet_embs' --force --experiment-name='lexsub-all-models' --run-name='coinco_xlnet_embs'
```

### 3.3. Web application
You could use command line interface to run Web application.
```shell script
# Run main server
lexsubgen-app run --host HOST 
                  --port PORT 
                  [--model-configs CONFIGS] 
                  [--start-ids START-IDS] 
                  [--start-all] 
                  [--restore-session]
``` 
**Example:**
```shell script
# Run server and serve models BERT and XLNet. 
# For BERT create server for serving model and substitute generator instantly (load resources in memory).
# For XLNet create only server.
lexsubgen-app run --host '0.0.0.0' 
                  --port 5000 
                  --model-configs '["my_cool_configs/bert.jsonnet", "my_awesome_configs/xlnet.jsonnet"]' 
                  --start-ids '[0]'

# After shutting down server JSON file with session dumps in the '~/.cache/lexsubgen/app_session.json'.
# The content of this file looks like:
# [
#     'my_cool_configs/bert.jsonnet',
#     'my_awesome_configs/xlnet.jsonnet',
# ]
# You can restore it with flag 'restore-session'
lexsubgen-app run --host '0.0.0.0' 
                  --port 5000 
                  --restore-session
# BERT and XLNet restored now
```
##### Arguments:
|Argument           |Default|Description                                                                                   |
|-------------------|-------|----------------------------------------------------------------------------------------------|
|`--help`           |       |Show this help message and exit                                                               |
|`--host`           |       |IP address of running server host                                                             |
|`--port`           |`5000` |Port for starting the server                                                                  |
|`--model-configs`  |`[]`   |List of file paths to the model configs.                                                      |
|`--start-ids`      |`[]`   |Zero-based indices of served models for which substitute generators will be created           |
|`--start-all`      |`False`|Whether to create substitute generators for all served models                                 |
|`--restore-session`|`False`|Whether to restore session from previous Web application run                                  |


### 4. FAQ
1. How to use gpu? - You can use environment variable CUDA_VISIBLE_DEVICES to use gpu for inference:
   ```export CUDA_VISIBLE_DEVICES='1'``` or ```CUDA_VISIBLE_DEVICES='1'``` before your command.
1. How to run tests? - You can use pytest: ```pytest tests```
