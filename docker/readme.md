## Create docker for running Megatron-LM and NeMo-toolkit
- Megatron-LM runtime: ```docker build - < Dockerfile-megatron -t alexgre/megatron```
- Nemo runtime: ```docker build - < Dockerfile-nemo -t alexgre/nemo```


## Create Singularity .sif 
> on Hipergator AI cluster, the env management is Singularity ( https://sylabs.io/singularity )
- we can directly create singularity .sif from docker hub
```singularity build megatron.sif docker://nvcr.io/nvidia/pytorch:21.02-py3```
```singularity build nemo.sif docker://nvcr.io/nvidia/nemo:1.0.0b3```
- we can create singularity using  definition files (note: you have to be root or sudo user)
```sudo singularity build nemo.sif nemo.def```