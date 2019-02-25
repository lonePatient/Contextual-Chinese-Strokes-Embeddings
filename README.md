# Contextual String Embeddings for Sequence Labeling

Implementation of the language model for Contextual chinese strokes Embeddings with PyTorch

This repository contains an implementation with PyTorch of the sequential model presented in the paper["Contextual String Embeddings for Sequence Labeling"](http://aclweb.org/anthology/C18-1139) by Alan Akbik et al. in 2018.

source code: [Falir](https://github.com/zalandoresearch/flair)

## Structure of the model

![]( https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/20190225231908.jpg)

![]( https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/20190225231917.jpg)


## Structure of the code

At the root of the project, you will see:

```
├── pyLM
|  └── callback
|  |  └── lrscheduler.py　　
|  |  └── trainingmonitor.py　
|  |  └── ...
|  └── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  └── dataset　　　
|  └── io　　　　
|  |  └── dataset.py　　
|  |  └── data_transformer.py　　
|  └── model
|  |  └── nn　
|  |  └── layers
|  └── output #save the ouput of model
|  └── preprocessing #text preprocessing 
|  └── train #used for training a model
|  |  └── trainer.py 
|  |  └── ...
|  └── test
|  |  └── embedding.py
|  └── utils # a set of utility functions
├── obtain_word_embedding.py
├── train_stroke_lm.py
```

## Dependencies

- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.0
- matplotlib
- pandas

## How to use the code

1. Prepare data, you can modify the `io.data_transformer.py` to adapt your data.
2. Modify configuration information in `pyLM/config/basic_config.py`(the path of data,...).
3. Run `train_stroke_lm.py` to training language model.
4. Run `obtain_word_embedding.py` to obtaining word embedding. 

## result

![]( https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/20190225231509.png)

![]( https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/20190225231516.png)
