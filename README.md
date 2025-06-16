# KGA

Our paper is available at [KGA: Privacy-Preserving Auditing of Sensitive Multi-hop Knowledge Membership in KG-RAG Systems](XXXX).

## Abstract

Knowledge Graph Retrieval-Augmented Generation (KG-RAG) systems integrate structured knowledge graph retrieval with text generation, offering enhanced reasoning capabilities and factual consistency in complex question answering tasks. However, their dependence on external knowledge sources introduces critical concerns regarding the unauthorized collection and use of personal or sensitive information embedded within these graphs. To ensure compliance with data protection regulations such as GDPR and to detect improper data usage, we propose the Knowledge Graph RAG Audit (KGA) framework, which enables users to audit whether their sensitive multi-hop knowledge has been included in KG-RAG systems, even in black-box settings without any prior knowledge of the system. It is effective across open-source and closed source KG-RAG systems and resilient to defense strategies. Experiments demonstrate that KGA achieves an improvement in AUC by 20.6% (compared to the best baseline), while maintaining strong performance under adversarial defenses. Furthermore, we analyze how the auditor’s knowledge of the target system affects performance, offering practical insights for privacy-preserving AI systems.


## Environmental installation

First, We suggest manually installing GPU-related libraries, selecting the version that is compatible with your CUDA version. You can find the compatible version at [torch install guide](https://pytorch.org/).

```
pip3 install torch torchvision torchaudio
```

Then, you should download the models in your customized file path or load with 
```
from transformers import AutoModelForCausalLM, AutoTokenizer
```
For example, you may have the following file structure.

```
|-- Model
    |-- llama-3.1-8b-instruct
    |   |-- config.json
    |   |-- generation_config.json
    |   |-- model-00001-of-00004.safetensors
    |   |-- model-00002-of-00004.safetensors
    |   |-- model-00003-of-00004.safetensors
    |   |-- model-00004-of-00004.safetensors
    |   |-- model.safetensors.index.json
    |   |-- special_tokens_map.json
    |   |-- tokenizer_config.json
    |   |-- tokenizer.json
```

Finally, to install other required packages, you can use following methods

Install from the `requirements.txt` file:

```
pip install -r requirements.txt
```

## About the data

You can also find the origin datasets here: [2WikiMultihopQA](https://huggingface.co/datasets/xanhho/2WikiMultihopQA) and [MetaQA](https://github.com/yuyuz/MetaQA).

For 2WikiMultihopQA, you can extract each question's corresponding multihop reasoning path in the `evidences` field of the dataset.

For MetaQA, you can also extract multihop reasoning path from file `kb.txt`

Please store your extracted multihop reasoning path in your customized data storage path, for example, `/Data`, in jsonl format. To be specific, each data sample in the jsonl file should have `path` field, for question generation:
```
{"path": [["Rubber Racketeers", "director", "Harold Young"], ["Harold Young", "place of birth", "Portland, Oregon"]]}

```


## Examples and illustrate

There are 2 main steps to run our experiment: preparation, audit and evaluation. Below is a brief introduction on how to run these Python script files, with detailed comments available for reference in each file.

### 1. preparation

In this section, we perform Neo4j backend installation,  and Insert your path, pre-processing on the datasets, and get the feature file.
#### Install and activate your local neo4j container

You can install your Neo4j graph storage backend with docker.

```
docker pull neo4j:5.19 #or any versions on command
```

and create your local Neo4j container:
```
docker run \
  --name neo4j-local \   #your container's name
  -p 7474:7474 -p 7687:7687 \
  -d \
  -e NEO4J_AUTH=neo4j/test123 \   #your neo4j backend's username and password
  -v $HOME/neo4j/data:/data \
  neo4j:5.19   #your neo4j version

```
You can then use your local browser with url : http://localhost:7474 and type in your username and password for your neo4j backend's virtualization.

#### Prepare your Multihop Questions dataset.

You need to evenly split your multi-hop path dataset into member and non-member subsets, ensuring that each subset contains an equal number of path for each hop length.

For your member paths dataset, you can insert these multihop paths into neo4j backend by running following codes:

```
from utils.utils import insert_member_paths
insert_member_paths(neo4j_uri= "your_neo4j_uri", neo4j_user = "your_neo4j_username", neo4j_password = "your_neo4j_password", jsonl_path= "your_member_path_jsonl", database="your_neo4j_container_name")

```
We have a Python script file `utils/Query_generator.py` for each path's corresponding Multihop Question's generation.

You can get your Multihop Questions dataset by simply running following codes:

```
from utils.Query_generator import QAPathGenerator

Generator = QAPathGenerator(model_id = "your_model_id", batch_size = "your_batch_size") #your local model or huggingface models, or OpenAI's GPT series.

await Generator.generate_from_jsonl(input_path = "your_jsonl_file_with_path", output_path = "your_output_question_dataset_path") #Input your jsonl file with multihop paths, and output a jsonl file with generated multihop questions.

```

#### Run a feature 
We prepared a Python script to generate the feature for your multihop question dataset. For virtualization, we maintain the true answer and generated answerfor each question. You can maintain the true answer when generating your multihop questions, by simply maintain the last entity of the reasoning path. 

You can run `run_feature.py` for the feature. If you are running feature for a dataset's member samples, you should set `label = 1`, and set `label = 0` for non-member samples.

If you are performing the experiment on 2WikiMultihopQA, you should set `target_dataset` to `2wiki` in line `cypher_response = await encoder.get_cypher_queries_async(question,target_dataset="target_dataset_here")`, similarly, for MetaQA dataset, set to `meta_qa`, and for OGBN-MAG dataset, set it to `OGBN-MAG`.

If you want to test different combination of Cypher Generation Model and Response Generation Model, you can set `encoder` to your choice of Cypher Generation Model, and `decoder` to your choice of Response Generation Model. By now, we support llama-3.1-8B-Instruct and OpenAI's GPT series.

By setting `path` and `output_path` to your multihop question dataset and feature output dataset, you can get the feature file of your member samples and non-member samples for auditing performing, in csv format.

### 2. audit and evaluation

We prepared a Jupytor notebook to perform shadow KG-RAG auditor training and evluate the auditor to target dataset.

You can simply set `posi_file` and `nega_file` to your member feature file and non-member feature file.

If you want to train your auditor on a dataset with different domain knowledge for testing, you can simply switch the `train_df` to this dataset.




