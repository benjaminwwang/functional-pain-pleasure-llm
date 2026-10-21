Code and data for my paper, uploaded as paper_manuscript.pdf

TRAINING

Final used CPT training data: 
training_data/cpt/cpt_verification/final_cpt.jsonl

Final used SFT training data: 
training_data/sft/sft_verification/final_sft.jsonl

These training data files were passed to the scripts in training_scripts for training

You can fairly easily reproduce my training by renting 2x A100 on runpod which is what I did, but if you have a reasonable request to want the actual 14 GB checkpoints, contact me

NOTE: the third "GhostGPT" persona was not used in my paper, so if you're just replicating my work, you can cut that out of the code for faster training

STEERING VECTOR EXTRACTION

Extract semantic pain vector from [Tagliabue et al. (2026)](https://arxiv.org/abs/2609.16247):
steering_vector_extraction/extraction_pain_vector_controlled.py (get the dataset from their github)

Extract functional-pain and functional-pleasure vectors introduced in my paper:
steering_vector_extraction/optimize_state_vectors.py

EVALUATION

The four evaluations in my paper were performed using the scripts in evaluation_scripts

GRADING

Responses and autograding scripts are in eval_responses
