# MEMM Part-of-Speech Tagging

This repository contains an implementation of a **Maximum Entropy Markov Model (MEMM)** for **Part-of-Speech (POS) tagging**, completed as part of a homework assignment. The project includes two trained models — a large, feature-rich model, and a smaller, simplified model — both evaluated on a held-out test set.

## Overview

The goal of this project is to train a MEMM-based POS tagger that assigns the correct tag to each word in a sentence using a set of handcrafted features. The tagger leverages previous tag information and contextual word features to make its predictions.

Two versions of the model were implemented:

- **Model A (Big model)**: Uses a wide variety of manually engineered features.
- **Model B (Small model)**: Uses a minimal feature set, designed to be more efficient and generalizable. There is no test set- more on that later. 

## Repository Structure
memm-pos-tagger/  
├── data/  
│ ├── comp1.words  
│ ├── comp2.words  
│ ├── train1.wtag  
│ ├── train2.wtag  
│ └── test1.wtag  
├── trained_models/  
│ ├── weights_1.pkl  
│ └── weights_2.pkl  
├── src/  
│ ├── generate_comp_tagged.py  
│ ├── inference.py  
│ ├── inference2.py  
│ ├── main.py  
│ ├── optimization.py  
│ ├── preprocessing.py  
│ ├── preprocessing2.py  
├── report.pdf  
├── comp_m1.wtag  
├── comp_m2.wtag  
└── README.md  

## Features Used

Model A (the larger model) includes features such as:

- Current word and previous predicted tag
- Word length and suffix patterns
- Capitalization (e.g., whether the word starts with an uppercase letter)
- Presence of hyphens or digits
- Word shape and token position
- Lexical patterns (e.g., tagging specific words more likely to appear as proper nouns)

Model B (the smaller model) uses a smaller, cleaner subset of features. It discards length-based features and focuses only on high-impact indicators like capitalization and hyphenation.

## Post-Processing

After decoding with Viterbi, a correction step is applied to fix common tagging mistakes, especially:

- Numbers and punctuation that were misclassified
- Words with hyphens or capital letters that were not tagged properly
- Tokens with low ambiguity that can be corrected deterministically based on context

This heuristic step helps improve the overall accuracy in predictable scenarios.

## Evaluation Results

Model	&nbsp;&nbsp;&nbsp; Accuracy (%)

Model A	 &nbsp;&nbsp;&nbsp; 93.98


Model B	&nbsp;&nbsp;&nbsp; 90.53
