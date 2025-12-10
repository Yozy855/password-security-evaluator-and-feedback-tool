Password Security Evaluator and Feedback Tool

Description:


Required Libraries and Installations:


How to generate files and run this program:

1) In order to train the ML password classifier, you must run the train_password_model.py, which will generate two files: password_model.pkl and password_vectorizor.pkl
   These files will then be used to load the model in another file.
2) To train the feedback model, run sft_feedback_llm.py. This takes larger computer power. I ran it on a 100GB GPU node but didn't utilize all of it.
   This file will generate a folder that contains adapter and tokenizer files, used for the new supervise fine-tuned model that will be loaded in the main program file.
4) Run full_program_sequence.py (CHANGE TO MAIN). This program will load all of the models and ask for input, which the user will get a classification for and potentially feedback.


Limitations:

Currently our program's accuracy is not where we would like it to be.
