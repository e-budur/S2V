# Turkish-Sentence Encoder with Quick-Thought Vectors
We are a team of graduate students working voluntarily at [Inzva Hacker Space](https://inzva.com) on building a model for Generic Sentence Encoder for the Turkish Language. Generic Sentence Encoders are used to let the computer understand what the user meant by their sentences. They are typically used in converstional assistants to understand the meaning of the user utterances. There are a number of recent research that are proposed for Generic Sentence Encoder for the English language as exemplified below.  

Universal Sentence Encoder (by Google AI)
https://ai.google/research/pubs/pub46808

Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning (by Microsoft Research)
https://openreview.net/forum?id=B18WgG-CZ

They leverage the Natural Language Inference (NLI) datasets to train their models as well as to evaluate their models as a downstream task. There are many NLI datasets that have been previously crowdsourced for the English language.  Examples in an NLI dataset can be seen below.

An example of entailment (similarity).
- Sentence 1: A senior is waiting at the window of a restaurant that serves sandwiches.
- Sentence 2: A person waits to be served his food.
 
An example of contradiction (dissimilarity).
- Sentence 1: A senior is waiting at the window of a restaurant that serves sandwiches.
- Sentence 2: A man is waiting in line for the bus.

Examples are taken from The Stanford Natural Language Inference (SNLI) Corpus (https://nlp.stanford.edu/projects/snli).

We want to build a Generic Sentence Encoder model for the Turkish. However, there are not any NLI datasets in the Turkish language due to the high cost of crowdsourcing. We want to leverage the new Amazon Translate Service to translate the NLI datasets available in the English language into the Turkish language as a cost effective solution and publish the translated datasets publicly to boost Turkish NLP works. 

# Experimentations

# Experiment #1

The results of the following experiments were obtained.

Experiment Name | Language      | Dataset               | Preprocessing                 |  Model # | Metric    | Value
--------------- | ------------- | --------------------  | ----------------------------- | -------- | --------- | -------------------
model1_task1    | English       | SICK                  |  Lowercase                    | Model1   | Pearson   | 0.8595461496671714
model2_task1    | English       | SICK                  |  Lowercase + sentencepiece    | Model2   | Pearson   | 0.8470309759444442
model1_task1    | English       | SICK                  |  Lowercase                    | Model1   | Spearman  | 0.7906599787429348
model2_task1    | English       | SICK                  |  Lowercase + sentencepiece    | Model2   | Spearman  | 0.7824858836725014
model1_task1    | English       | SICK                  |  Lowercase                    | Model1   | MSE       | 0.2669741153404767
model2_task1    | English       | SICK                  |  Lowercase + sentencepiece    | Model2   | MSE       | 0.28983657549983965
model1_task1    | English       | SNLI                  |  Lowercase                    | Model1   | Accuracy  | 71.32%
model2_task1    | English       | SNLI                  |  Lowercase + sentencepiece    | Model2   | Accuracy  | 69.63%
model1_task1    | English       | MultiNLI (matched)    |  Lowercase                    | Model1   | Accuracy  | 59.69%
model1_task1    | English       | MultiNLI (mismatched) |  Lowercase                    | Model1   | Accuracy  | 60.84%
model2_task1    | English       | MultiNLI (matched)    |  Lowercase + sentencepiece    | Model2   | Accuracy  | 59.04%
model2_task1    | English       | MultiNLI (mismatched) |  Lowercase + sentencepiece    | Model2   | Accuracy  | 60.27%
model1_task2    | Turkish       | SICK-MT-TR            |  Lowercase                    | Model1   | Pearson   | 0.7767414617451377
model2_task2    | Turkish       | SICK-MT-TR            |  Lowercase + sentencepiece    | Model2   | Pearson   | 0.8076206267469718
model1_task2    | Turkish       | SICK-MT-TR            |  Lowercase                    | Model1   | Spearman  | 0.7042856789726142
model2_task2    | Turkish       | SICK-MT-TR            |  Lowercase + sentencepiece    | Model2   | Spearman  | 0.7348411904626335
model1_task2    | Turkish       | SICK-MT-TR            |  Lowercase                    | Model1   | MSE       | 0.40369925427270614
model2_task2    | Turkish       | SICK-MT-TR            |  Lowercase + sentencepiece    | Model2   | MSE       | 0.3561044127205771
model1_task2    | Turkish       | SNLI-MT-TR            |  Lowercase                    | Model1   | Accuracy  | 62.41%
model2_task2    | Turkish       | SNLI-MT-TR            |  Lowercase + sentencepiece    | Model2   | Accuracy  | 64.38%
model1_task2    | Turkish       | MultiNLI (matched)    |  Lowercase                    | Model2   | Accuracy  | 50.84%
model1_task2    | Turkish       | MultiNLI (mismatched) |  Lowercase                    | Model2   | Accuracy  | 51.61%
model2_task2    | Turkish       | MultiNLI (matched)    |  Lowercase + sentencepiece    | Model2   | Accuracy  | 59.04%
model2_task2    | Turkish       | MultiNLI (mismatched) |  Lowercase + sentencepiece    | Model2   | Accuracy  | 53.36%

Not all results of the models on the MultiNLI and XNLI datasets were included in the table since they need some more analysis.

# Announcements

:hatching_chick: (2018-10-26) Our research has been awarded free GPU resource by [TRUBA](https://www.truba.gov.tr)  which is the Turkish Science e-Infrastructure that is provided by The Scientific and Technological Research Council of Turkey ([TUBITAK](https://www.tubitak.gov.tr/en)).

:hatching_chick: (2019-01-26) Our research has been granted [AWS Research Credits](https://twitter.com/ebudur/status/1090301816183685120)   to translate the most commong NLI datasets into Turkish and evaluate our resulting NLI models.

:dart: (2019-03-31) We have applied for the AWS Research Credits again to translate all available NLI and Textual Entailment datasets into Turkish.

- - - -

# References
This repository is based on the implementation of the following paper in [this](https://github.com/lajanugen/S2V) Github repository which belongs to the following paper.

Lajanugen Logeswaran, Honglak Lee, 
[An efficient framework for learning sentence representations](https://arxiv.org/pdf/1803.02893.pdf). In ICLR, 2018.
