# LexSubGen
Lexical Substitution Framework

This repository contains the code to reproduce the results from the paper:

Arefyev Nikolay, Sheludko Boris, Podolskiy Alexander, Panchenko Alexander, 
["Always Keep your Target in Mind: Studying Semantics and Improving Performance of Neural Lexical Substitution"](https://www.aclweb.org/anthology/2020.coling-main.107/), 
Proceedings of the 28th International Conference on Computational Linguistics, 2020


## Installation
Clone LexSubGen repository from github.com.
```shell script
git clone https://github.com/Samsung/LexSubGen
cd LexSubGen
```

### Setup anaconda environment
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
<!--
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
-->
### Setup Web Application
**If you do not plan to use the Web Application, skip this section and [go to the next](#13-install-lexsubgen-library)!**
1. Download and install [NodeJS and npm](https://www.npmjs.com/get-npm).
2. Run script for install dependencies and create build files.
```shell script
bash web_app_setup.sh
```

### Install lexsubgen library
```shell script
python setup.py install
```
<!---
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
-->
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
            <td>OOC</td>
            <td>44.65</td>
            <td>16.82</td>
            <td>12.83</td>
            <td>18.36</td>
            <td>46.3</td>
            <td>19.58</td>
            <td>15.03</td>
            <td>12.99</td>
        </tr>
        <tr>
            <td>C2V</td>
            <td>55.82</td>
            <td>7.79</td>
            <td>5.92</td>
            <td>11.03</td>
            <td>48.32</td>
            <td>8.01</td>
            <td>6.63</td>
            <td>7.54</td>
        </tr>
        <tr>
            <td>C2V+embs</td>
            <td>53.39</td>
            <td>28.01</td>
            <td>21.72</td>
            <td>33.52</td>
            <td>50.73</td>
            <td>29.64</td>
            <td>24.0</td>
            <td>21.97</td>
        </tr>
        <tr>
            <td>ELMo</td>
            <td>53.66</td>
            <td>11.58</td>
            <td>8.55</td>
            <td>13.88</td>
            <td>49.47</td>
            <td>13.58</td>
            <td>10.86</td>
            <td>11.35</td>
        </tr>
        <tr>
            <td>ELMo+embs</td>
            <td>54.16</td>
            <td>32.0</td>
            <td>22.2</td>
            <td>31.82</td>
            <td>52.22</td>
            <td>35.96</td>
            <td>26.62</td>
            <td>23.8</td>
        </tr>
        <tr>
            <td>BERT</td>
            <td>54.42</td>
            <td>38.39</td>
            <td>27.73</td>
            <td>39.57</td>
            <td>50.5</td>
            <td>42.56</td>
            <td>32.64</td>
            <td>28.73</td>
        </tr>
        <tr>
            <td>BERT+embs</td>
            <td>53.87</td>
            <td>41.64</td>
            <td>30.59</td>
            <td>43.88</td>
            <td>50.85</td>
            <td>46.05</td>
            <td>35.63</td>
            <td>31.67</td>
        </tr>
        <tr>
            <td>RoBERTa</td>
            <td>56.74</td>
            <td>32.25</td>
            <td>24.26</td>
            <td>36.65</td>
            <td>50.82</td>
            <td>35.12</td>
            <td>27.35</td>
            <td>25.41</td>
        </tr>
        <tr>
            <td>RoBERTa+embs</td>
            <td>58.74</td>
            <td>43.19</td>
            <td>31.19</td>
            <td>44.61</td>
            <td>54.6</td>
            <td>46.54</td>
            <td>36.17</td>
            <td>32.1</td>
        </tr>
        <tr>
            <td>XLNet</td>
            <td>59.12</td>
            <td>31.75</td>
            <td>22.83</td>
            <td>34.95</td>
            <td>53.39</td>
            <td>38.16</td>
            <td>28.58</td>
            <td>26.47</td>
        </tr>
        <tr>
            <td>XLNet+embs</td>
            <td>59.62</td>
            <td>49.53</td>
            <td>34.9</td>
            <td>47.51</td>
            <td>55.63</td>
            <td>51.5</td>
            <td>39.92</td>
            <td>35.12</td>
        </tr>
    </tbody>
</table>


### Results reproduction
Here we list XLNet reproduction commands that correspond
to the results presented in the table above. Reproduction commands for all models you can 
find in ```scripts/lexsub-all-models.sh``` Besides saving to the 'run-directory' 
all results are saved using mlflow. To check them you can run ```mlflow ui``` in LexSubGen 
directory and then open the web page in a browser. 

Also you can use pytest to check the reproducibility. But it may take a long time:
```shell script
pytest tests/results_reproduction
```
* #### XLNet:
XLNet Semeval07: 
```shell script
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/xlnet.jsonnet --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet --run-dir='debug/lexsub-all-models/semeval_all_xlnet' --force --experiment-name='lexsub-all-models' --run-name='semeval_all_xlnet'
```

XLNet CoInCo: 
```shell script
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/xlnet.jsonnet --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet --run-dir='debug/lexsub-all-models/coinco_xlnet' --force --experiment-name='lexsub-all-models' --run-name='coinco_xlnet'
```

XLNet with embeddings similarity Semeval07:
```shell script
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/xlnet_embs.jsonnet --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet --run-dir='debug/lexsub-all-models/semeval_all_xlnet_embs' --force --experiment-name='lexsub-all-models' --run-name='semeval_all_xlnet_embs'
```

XLNet with embeddings similarity CoInCo:
```shell script
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/xlnet_embs.jsonnet --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet --run-dir='debug/lexsub-all-models/coinco_xlnet_embs' --force --experiment-name='lexsub-all-models' --run-name='coinco_xlnet_embs'
```

## Word Sense Induction Results
<table>
    <thead>
        <tr>
            <th rowspan=2><b>Model</b></th>
            <th colspan=1><b>SemEval 2013</b></th>
            <th colspan=1><b>SemEval 2010</b></th>
        </tr>
        <tr>
            <th>AVG</th>
            <th>AVG</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>XLNet</td>
            <td>33.4</td>
            <td>52.1</td>
        </tr>
        <tr>
            <td>XLNet+embs</td>
            <td>37.3</td>
            <td>54.1</td>
        </tr>
    </tbody>
</table>

To reproduce these results use 2.3.0 version of transformers and the following command:
```shell script
bash scripts/wsi.sh
```

### Web application
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


### FAQ
1. How to use gpu? - You can use environment variable CUDA_VISIBLE_DEVICES to use gpu for inference:
   ```export CUDA_VISIBLE_DEVICES='1'``` or ```CUDA_VISIBLE_DEVICES='1'``` before your command.
1. How to run tests? - You can use pytest: ```pytest tests```
