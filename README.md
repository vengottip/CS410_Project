

# CS410 Project: TweetInsight Explorer for Technology

## Overview

Our system, the TechSentiment Insight Analyzer, developed for the CS410 final project, empowers users to gain insights into sentiments in the technology industry and leading tech companies. Users input specific queries, and the system displays relevant tweets, identifies and presents the topics expressed in the relevant tweets and frequent terms used in the topics, offering an overview of the public discourse related to the query. Our system also performs sentiment analysis to show distribution of positive, negative, and neutral sentiments in the relevant tweets.


## Dependencies 
Required packages are listed in requirements.txt



```
sleep_train:
    _target_: dpr.data.biencoder_data.JsonQADataset
    file: "../../../../data/training/sleep-train_aug2.json"
```



## Dataset Creation:

To construct our dataset of tweets, we collected the most recent posts using hashtags associated with major tech companies. 

We utilized the ntscraper library for scraping purposes. Following that, we conducted data cleaning to remove URLs, emojis, symbols, and mentions of Twitter users to preserve user privacy. In total, our dataset consists of 21,600 tweets.



## Text Retrieval:

We use PyTerrier which is a comprehensive and scalable toolkit for information retrieval. We index the content (text) and document number for faster retrieval later. After comparing different weighting models such as BM25 and DirichletLM, we choose BM25 for its effectiveness and robustness. Then we obtain the result from queries and list the most relevant contents to the UI.
- Main configuration: conf/extractive_reader_train_cfg.yaml <br />
It has been set using hf_BioASQ as encoder in name of encoder parameter. Set the number of gpu for n_gpu parameter if GPU can be used for training. Otherwise, n_gpu should be set to 0, and no_cuda set to True.
  
- encoder configuration:  <br />
conf/encoder/hf_BioASQ.yaml is the one used in this study, no need to change the configuration.
  
- train configuration: conf/train/extractive_reader_default.yaml  <br />
Set the hyperparameter for reader. The batch_size is set to 3 due to memory limits. A larger batch size may help improving accuracy, but unlike retriever training which requires contrastive loss, reader model trains each example independently without using in-batch pairs. As a result, increasing the batch size only helps to reduce gradient noise without providing other additional benefits. Standard size that generally works well is 16 (used by the authors of SleepQA) or 32. One can change accordingly if GPU memory is sufficient.

After setting the configs, to train reader, run train_extractive_reader.py. The outputs will be saved in DPR-main/outputs/yyyy-mm-dd. 



## Files: 

xxx.py: it t

