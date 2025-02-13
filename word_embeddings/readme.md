# PoseC3D (Including Semantic Information via Word Embeddings for Skeleton-based Action Recognition)

## Abstract

Effective human action recognition is widely used for cobots in Industry 4.0 to assist in assembly tasks.
However, conventional skeleton-based methods often lose keypoint semantics, limiting their effectiveness in complex interactions.
In this work, we introduce a novel approach to skeleton-based action recognition that enriches input representations by leveraging word embeddings to encode semantic information.
Our method replaces one-hot encodings with semantic volumes, enabling the model to capture meaningful relationships between joints and objects.
Through extensive experiments on multiple assembly datasets, we demonstrate that our approach significantly improves classification performance, and enhances generalization capabilities by simultaneously supporting different skeleton types and object classes.
Our findings highlight the potential of incorporating semantic information to enhance skeleton-based action recognition in dynamic and diverse environments.


## Results and Models

### IKEA ASM

|        data      | frame sampling strategy | dimension | vector aggregation |   backbone   | mAcc | top1 | epochs | testing protocol |     config    |
| :--------------: | :---------------------: | :-------: | :----------------: | :----------: | :--: | :--: | :----: | :--------------: | :-----------: |
|      joints      |       uniform 54        |     20    |         sum        | SlowOnly-R50 | 41.6 | 74.7 |   240  |     10 clips     |   [config](/configs/skeleton/posec3dplus/ikea_without_objects.py)  |
| joints + objects |       uniform 54        |     16    |         sum        | SlowOnly-R50 | 53.4 | 82.1 |   240  |     10 clips     |   [config](/configs/skeleton/posec3dplus/ikea_with_objects.py)  |

### ATTACH

|        data      | frame sampling strategy | dimension | vector aggregation |   backbone   | mAcc | top1 | epochs | testing protocol |     config    |
| :--------------: | :---------------------: | :-------: | :----------------: | :----------: | :--: | :--: | :----: | :--------------: | :-----------: |
|      joints      |       uniform 54        |     20    |         sum        | SlowOnly-R50 | 44.9 | 55.1 |   60   |     10 clips     |   [config](/configs/skeleton/posec3dplus/attach_without_objects.py)  |
| joints + objects |       uniform 54        |     24    |         sum        | SlowOnly-R50 | 55.1 | 61.5 |   60   |     10 clips     |   [config](/configs/skeleton/posec3dplus/attach_without_objects.py)  |


## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

For training with your custom dataset, you can refer to [Custom Dataset Training](/configs/skeleton/posec3d/custom_dataset_training.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```


## Training with multiple datasets
Processing for only two datasets is implemented (IKEA ASM and ATTACH). The config file must specify two paths to word embeddings (arg: embedding_path2).
The samples from the datasets must comtain 'data1' and 'data2' in the 'frame_dir' value, so that the sample can be assigned to the correct dataset.
The following files are used for [prediction](/mmaction/models/heads/multi_head.py) and [evaluation](mmaction/evaluation/metrics/acc_metric_two_datasets.py). 
When using different datasets these files may also need to be adapted.



## File Explanation for Word Embeddings

| File Name                        | Description                                                                                              |
|----------------------------------|----------------------------------------------------------------------------------------------------------|
| **`download_embeddings.bash`**   | Executes download and extraction of fastText (300-d) Word Embeddings.                                    |
| **`create_vocabulary.py`**       | Responsible for collecting a few hundred to many thousand words and their corresponding word embeddings. |
| **`encoder.py`**                 | Dimension reduction of word embeddings with neural encoder.                                              |
| **`pca.py`**                     | Dimension reduction of word embeddings with PCA.                                                         |
| **`write_embeddings_to_pkl.py`** | Collects embeddings from relevant words for action recognition.                                          |
| **`visualization.py`**           | Performs visualization of word embeddings based on cosine similarity.                                    |
| **`loss_computation.py`**        | Calculates the loss for the reduced word embeddings based on cosine similarity.                          |


## Steps to create Word Embeddings for training PoseConv3D
1. Download word embeddings via download_embeddings.bash
2. Execute create_vocabulary.py to generate a small vocabulary with relevant words.
3. Perform dimension reduction
   1. PCA or
   2. Encoder
4. Write the reduced word embeddings to a .pkl file and specify the path in the [config](/configs/skeleton/posec3dplus/ikea_without_objects.py)

(Some parameters of the files need to be adapted depending on the use-case. Please refer to the documentation within the respective files.)