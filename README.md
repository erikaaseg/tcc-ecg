# ECG Classifier with XAI
Um modelo de Deep Learning com explicação para classificação de Eletrocardiogramas


## Pré-Requisitos

#### 1. Ensure you have Python 3.8+ installed
```
python --version
```

#### 2. Install required packages (if not already installed)
```
pip install torch torchvision pytorch-lightning
pip install grad-cam albumentations
pip install matplotlib numpy scipy pandas
```

## Conjunto de dados

Para este notebook, foram utilizados os dados "ECG Images dataset of Cardiac Patients", disponibilizado em https://data.mendeley.com/datasets/gwbz3fsgp8/2.

## Arquivos

* dataset.py: encapsulamento da leitura do conjunto de dados
* ecgclassifier_model.py: encapsulamento do modelo de classificação do ECG
* train.ipynb: encapsulamento do treinamento do modelo de classificação, gravando-o em arquivo
* gradcam.ipynb: estudo da interpretablilidade do modelo
* pertubation_test.ipynb: teste de perturbação em cima do Score-CAM
