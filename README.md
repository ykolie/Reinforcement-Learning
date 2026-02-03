# Practice fine-tuning the Llama-2 model using RLHF

## Overview
Large language models (LLMs) are trained on human-generated text, but additional methods are needed to align an LLM with human values and preferences.

Reinforcement Learning from Human Feedback (RLHF) is a method for aligning LLMs with human values and preferences. It's a branch of machine learning in which a model learns by acting, receiving feedback, and readjusting itself to maximise future feedback. 

In reinforcement fine-tuning (RFT), that reward signal comes from a custom grader that you define for your task. For every prompt in your dataset, the platform samples multiple candidate answers, runs your grader to score them, and applies a policy-gradient update that nudges the model toward answers with higher scores. This cycle—sample, grade, update—continues across the dataset (and successive epochs) until the model reliably optimizes for your grader’s understanding of quality. The grader encodes whatever you care about—accuracy, style, safety, or any metric—so the resulting fine-tuned model reflects those priorities and you don't have to manage reinforcement learning infrastructure.

1. **Gain a High-Level Understanding of RLHF**: Explore RLHF training processs and the importance of aligning LLMs with human values and preferences.<br />
    **Supervised Fine Tuning :** {input text, summary}<br />
    **RLHF :**                   {input text, summary 1, summary 2, human preference}

<img width="1535" height="780" alt="1_2" src="https://github.com/user-attachments/assets/1c602499-e8cf-4c20-b637-e375e0c27b8a" />

2. **Explore Datasets for RLHF Training**: Deep Dive into the "preference" and "prompt" datasets crucial for RLHF training.

<img width="892" height="634" alt="2_1" src="https://github.com/user-attachments/assets/25365224-6bcc-46a0-8682-ddcd5c3e4687" />


3. **Google Cloud Pipeline Components Library Fine Tune**: Practice fine-tuning the Llama 2 model using RLHF and the open-source Google Cloud Pipeline Components Library.

<img width="991" height="811" alt="3_1" src="https://github.com/user-attachments/assets/8967f0f9-775c-4ace-8bfb-b0f9412e806e" />


4. **Evaluate the Model**: Assess the tuned LLM against the original base model using loss curves and the "Side-by-Side (SxS)" method.

## About
- Explore the two datasets: “preference” and “prompt” datasets.
- Use the open source Google Cloud Pipeline Components Library, to fine-tune the Llama 2 model with RLHF.
- Assess the tuned LLM against the original base model by comparing loss curves and using the “Side-by-Side (SxS)” method.

Sources:
1. https://learn.deeplearning.ai/courses/reinforcement-learning-from-human-feedback/information
