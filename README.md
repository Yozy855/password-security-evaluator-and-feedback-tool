Password Security Evaluator and Feedback Tool

Description:

This github repository contains scripts that altogether form a program that utilitzes existing solutions like a classifier model, but also additional checks
and features for the problem of insecure passwords that pass baseline requirements. 
   
   Hybrid ML + LLM password evaluation framework with an integrated dictionary “attack”​

      - ML model classifies password strength from real patterns, not rules​

      - If words/phrases within the password are detected in common passwords/names dataset, flagged as not secure enough​

      - LLM interprets models rating and provides human friendly feedback​

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Required Libraries and Installations:

Python 3.11.9

pip install pandas

pip install torch

pip install transformers

pip install sklearn

pip install joblib

pip install peft

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

How to generate new model files and run this program:

1) In order to train the ML password classifier, you must run the train_password_model.py, which will generate two files: password_model.pkl and password_vectorizor.pkl
   These files will then be used to load the trained model in another file.
2) To train the feedback model, run sft_feedback_llm.py. This takes larger computer power. I ran it on a 100GB GPU node but didn't utilize all of it.
   This file will generate a folder that contains adapter and tokenizer files, used for the new supervise fine-tuned model that will be loaded in the main program file.
4) Run full_program_sequence.py. This program will load all of the models and ask for input, which the user will get a classification for and potentially feedback.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Limitations:

Currently our program's accuracy is not where we would like it to be. With the classifier being at 86% overall accuracy. Training it on more data and altering some parameters might contribute to a higher accuracy. As well for the feedback model, which is not as specific about why a password is weak and how to improve it.
