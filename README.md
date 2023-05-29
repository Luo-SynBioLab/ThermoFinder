# ThermoFinder: A sequence-based thermophilic proteins prediction framework

## 1. Introduction
These files contain source code for ThermoFinder. And established datasets in this article are also provided in [huggingface](https://huggingface.co/datasets/HanselYu/ThermoSeqNet).
ThermoFinder is a python implementation of the model.

## 2. Installation
```
python = 3.8.13
```
You could configure enviroment by running this
```
pip install -r requirment.txt
```

## 3. Requirments
In order to run successfully, the generation of embedded vectors requires GPU. We utilized an NVIDIA GeForce RTX 3080 with 10018MiB to embed protein sequences to a vector.
Other hardware equipments are not necessary.

## 4. Usage
For ThermoFinder, you could run Fused_model_XX.py file,  model training and prediction are all implemented.

## 5. Contact
If you have any question, you could contact yuhanid147@gmail.com.
