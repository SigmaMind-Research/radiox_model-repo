# Introduction of RadioX model
In our project, we utilize X-ray images, including any available patient history data, as our input. We employ a Convolutional Neural Network (CNN) to extract informative features from these X-ray images. Specifically, we focus on capturing disease-state-style information. Traditionally, many approaches used detected disease embeddings as the input for text generation networks. However, this approach may inadvertently exclude diseases that are not identified.

To address this issue and ensure consistency between detected diseases and generated X-ray reports, we've implemented an interpreter module. This interpreter plays a crucial role in verifying the accuracy of the generated X-ray reports.
# Data we used for experiments
We use two datasets for experiments to validate our method: 

  - [OpenI](https://openi.nlm.nih.gov/)
  - [MIMIC](https://physionet.org/content/mimiciii-demo/1.4/)
  

# Performance on two datasets
| Datasets | Methods                        | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L |
| -------- | ------------------------------ | ------ | ------ | ------ | ------ | ------ | ------- |
| Open-I   | Single-view                    | 0.463  | 0.310  | 0.215  | 0.151  | 0.186  | 0.377   |
|          | Multi-view                     | 0.476  | 0.324  | 0.228  | 0.164  | 0.192  | 0.379   |
|          | Multi-view w/ Clinical History | 0.485  | 0.355  | 0.273  | 0.217  | 0.205  | 0.422   |
|          | Full Model (w/ Interpreter)    | **0.515**  | **0.378**  | **0.293**  | **0.235**  | **0.219**  | **0.436**   |
| MIMIC    | Single-view                    | 0.447  | 0.290  | 0.200  | 0.144  | 0.186  | 0.317   |
|          | Multi-view                     | 0.451  | 0.292  | 0.201  | 0.144  | 0.185  | 0.320   |
|          | Multi-view w/ Clinical History | 0.491  | 0.357  | 0.276  | 0.223  | 0.213  | 0.389   |
|          | Full Model (w/ Interpreter)    | **0.495**  | **0.360**  | **0.278**  | **0.224**  | **0.222**  | **0.390**   |

# Environments for running codes
   
   - we have used kaggle notebook with A100 GPU

# This repo has two methods for model deploment 
    
    - Using Gardio App
    - troch-serve