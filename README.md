# GatorTron
all scripts used in gatortron project

## Project
Rapid adoption of electronic health record (EHR) systems have made large collections of real-world EHR data available for research. Nevertheless, much of the critical information about a patient, such as family history, drug adverse events, and social, behavioral, and environmental determinants of health, is well-documented only in narrative clinical text as opposed to structured EHR data. Clinical concept extraction through named-entity recognition (NER) is the key technology to unlock the rich patient characteristics buried in unstructured clinical text to support downstream applications that rely on structured data.  
 developPosit_chem
Recent advancements in deep learning, especially the Transformer architectures, have emerged as the current state-of-the-art Natural Language Processing (NLP). The performance of Transformer models heavily depends on the size and data domain of the training corpus used to generate pre-trained language models. In this project, we trained the largest clinical language model to date, Gatortron. Gatortron was trained using clinical notes available from the University of Florida Health system that cover more than 1 million patients, hundreds of times larger than the largest existing pretrained transformer-based language models’ corpora. In addition to using a domain-specific vocabulary, the model was trained by leveraging Nvidia’s Megatron transformer-based language modeling framework for model-parallel (tensor and pipeline) and accelerated multi-node pre-training across a DGX SuperPOD with over 1,000 A100 GPUs.
 
An evaluation of Gatortron using a de-identification task (i.e., detect and remove 18 personal identifiers such as names and birth dates from protected health information) showed that the newly trained Gatortron language model, achieved state-of-the-art performance.


## Workflow
![workflow](resources/gatorTron_workflow.png)

- pretraining GatorTron LM 
- fine tuning pretrained GatorTron models on downstream clinical tasks like 2010i2b2 NER


## Pretraining
- leverage the Megatron-LM (v2.2)
- distributed pretrained models on 120+ nodes with 900+ GPUs
- pretrainined dataset: UF notes + pubmed + wiki (500GB+ text; 100+ billion words)
- Gatortron-base (345m parameters; L:24 H:1280 A:12)
- Gatortron-medium (3.9b parameters; L:48 H:2560 A: 40)
- Gatortron-large (8.9b parameters; L:56 H:3560 A: 56) (model architecture choice is based on https://arxiv.org/abs/2006.12467)


## Fine-tuning
- leverage the NeMo toolkit
- NER benchamark on 2010i2b2 2012i2b2, and 2018 n2c2
- RE (relation extraction) benchmark on 2018 n2c2, chemprot
- QA benchmark on emrQA (relation + medication), BioASQ-7b-factoid
- NLI benchmark on MedNLI
- STS benchmark on clinical STS from 2019 n2c2 challenge


## Model availability
- the gatortron-345m model is publicly available at https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/gatortron_og

## Hugging Face releases
- GatorTron-base model, 345 million: https://huggingface.co/UFNLP/gatortron-base
- GatorTron-mediym model, 3.9 billion: https://huggingface.co/UFNLP/gatortron-medium

## How to use
The GatorTron series models are **Encoder only architecture**, therefore, the models aim to solve **natural language understanding** problems like NER, RE, MRC etc. The model is **not suitable for generative tasks**.

### huggingface
```python
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer, AutoConfig

tokinizer= AutoTokenizer.from_pretrained('UFNLP/gatortron-medium')
config=AutoConfig.from_pretrained('UFNLP/gatortron-medium')
mymodel=AutoModel.from_pretrained('UFNLP/gatortron-medium')

encoded_input=tokinizer("Bone scan:  Negative for distant metastasis.", return_tensors="pt")
encoded_output = mymodel(**encoded_input)

# then you can feed encoded_output to downstream task layers for different usecases e.g., NER, RE, MRC etc.
```


## Disclaimer
Although we did extension data checking during the training process to ensure the compliance of the trained model to the best of our ability, due to the complexity of the data and the diversity of language model usage scenarios, we cannot guarantee that the model will generate correct and reasonable output in all scenarios. Please be aware that there is still a risk of the model producing problematic outputs. We will not be responsible for any risks and issues resulting from misuse, misguidance, illegal usage, and related misinformation, as well as any associated data security concerns.


## Cite
please cite our paper:
```
Yang X, Chen A, PourNejatian N, Shin HC, Smith KE, Parisien C, Compas C, Martin C, Costa AB, Flores MG, Zhang Y, Magoc T, Harle CA, Lipori G, Mitchell DA, Hogan WR, Shenkman EA, Bian J, Wu Y†. A large language model for electronic health records. Npj Digit Med. Nature Publishing Group; . 2022 Dec 26;5(1):1–9. https://www.nature.com/articles/s41746-022-00742-2
```
https://www.nature.com/articles/s41746-022-00742-2
