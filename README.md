# Tailoring Adaptive-Zero-Shot Retrieval and Probabilistic Modelling for Psychometric Data

This repository contains the code related to the work ``Tailoring Adaptive-Zero-Shot Retrieval and Probabilistic Modelling for Psychometric Data'', accepted as *poster* at *Symposium of Applied Computing (SAC'25)*.

The workflow pipeline is divided into two main components. The first is a retrieval component that identifies the most relevant Reddit posts from each user to answer a psychological questionnaire. The second is a predictive model called PoissonBERT that accurately predicts questionnaire scores.

<img src="https://github.com/Fede-stack/PoissonBERT/blob/main/images/workflow.png" alt="" width="300">

* In `/retrieval/adaptive_retrieval_functions.py`, there are functions to implement adaptive retrieval, particularly for the adaptive proximity structure.
* In `/retrieval/semantic_retrieval_functions.py`, there is the implementation of semantic retrieval.
* In `/model/PoissonBERT.py`, there is the architecture implemented in the work.
