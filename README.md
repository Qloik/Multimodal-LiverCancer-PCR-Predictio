# Treatment Efficacy Prediction for Liver Cancer Using Multi-timepoint MRI

* A Deep Learning Project, Tongji University (2024)*

## Writer

Yifan Fu

## Project Overview

A deep learning approach to predict neoadjuvant therapy efficacy for liver cancer patients using:
* Multi-sequence MRI images from multiple timepoints
* Clinical blood test data (tumor markers)
* Deep neural networks for medical image analysis

* <img width="849" alt="image" src="https://github.com/user-attachments/assets/c2301921-2512-44f2-896d-54713bb48353" />

## Dataset

* 6,600 MRI images from 113 liver cancer patients (provided by Zhongshan Hospital, Fudan University)
* Data preprocessing:
   * Data cleaning and format standardization
   * Manual tumor lesion annotation on multi-modal MRI images
   * Integration of clinical blood test records
* Physician-annotated treatment outcomes

* ![image](https://github.com/user-attachments/assets/65a5b5df-1f12-4519-8ff1-ecc2b42db75c)

## Technical Approach

### Model Evolution

1. **Baseline Model (ResNet18)**
  * Single-modal, single-timepoint prediction
  * Initial performance evaluation and analysis

2. **ResNet_PCR_single**
  * Enhanced single-modal architecture
  * Integration of:
    * Clinical blood test data
    * Multi-timepoint information
  * Comparative experiments and interpretability analysis

3. **ResNet_PCR_multi**
  * Multi-modal fusion architecture
  * Addressing class imbalance
  * Ablation studies on modal performance
  * Clinical application value analysis

  * ![image](https://github.com/user-attachments/assets/e281becd-41dd-4f93-9414-b1af42045792)

  * ![image](https://github.com/user-attachments/assets/df700903-69f1-4d94-bde8-4a3012a78d94)



4. **ResNet_PCR_relation (Final Model)**
  * Novel multi-timepoint relation module
  * Features:
    * Similarity and difference computation between timepoints
    * Temporal trajectory modeling of tumor evolution
    * Weighted feature fusion strategy
   
    * ![image](https://github.com/user-attachments/assets/659800ff-9feb-43c4-8d48-97c613a7e274)


### Implementation Details
* Deep learning framework: PyTorch
* Multi-modal data integration pipeline
* Time-series medical image analysis
* Performance metrics focused on clinical relevance


## Results & Analysis

* Comparative analysis across model versions
* Ablation studies on different modalities
* Clinical interpretation of predictions
* Model interpretability analysis

* ![image](https://github.com/user-attachments/assets/03ddf407-a0b5-4591-8014-30e36485c8e6)

* ![image](https://github.com/user-attachments/assets/c9270e55-c912-42c8-950e-64ace7a3ab46)

* ![image](https://github.com/user-attachments/assets/176a54c7-766b-4285-a1ed-c1e1bec967b2)

* ![image](https://github.com/user-attachments/assets/74819265-690a-46e0-8730-b5aef1e8393b)



**Data Access:**  
Link: https://pan.baidu.com/s/1FWV4pNIZ9bYIajUQOjq9rg?pwd=op7x  
Password: op7x

**Keywords:** Deep Learning, Medical Image Analysis, Multi-modal Fusion, Multi-timepoint MRI, Treatment Outcome Prediction

## Acknowledgements

Special thanks to Zhongshan Hospital, Fudan University for providing the dataset and clinical expertise.






