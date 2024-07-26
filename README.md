# IAMGPLA
Interaction-Aware Multi-Graph Neural Network Integration Strategy for Protein-Ligand Binding Affinity Prediction
    
The benchmark dataset can be found in ./data/. The IAMGPLA model is available in ./src/. And the result will be generated in ./result/ and ./model/. See our paper for more details.
    
You can download datasets from the following link: 
[PDBbind](http://pdbbind.org.cn/)、[CSAR-NRC HiQ](http://www.csardock.org/)
## Requirements：
	python=3.8.0
	pytorch=1.12.1
	numpy=1.24.3
	dgl=1.1.1.cu113
	rdkit-pypi==2022.9.5
	pandas==2.0.3
	scikit-learn==1.3.1
	matplotlib==3.4.3
## Create environment with GPU-enabled version:
	#Run the commandline
	conda env create -f environment.yml
	conda activate IAMGPLA
## Data process：
graph construction

	#Run the commandline
	cd ./src/
	python process.py		
## Train:
to train your own model

	#Run the commandline
	cd ./src/
	python train.py
## Test:
to test our model

	#Run the commandline
	cd ./src/
	python predict.py
