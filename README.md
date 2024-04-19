# Chatbot for Duke AI MEng Program

## Introduction
This project aims to build a chatbot for the Duke AI MEng Program. The chatbot is designed to help students with questions about the program and provide useful information. The chatbot is built on a pipeline that consists of a Retriever-Reader model, a Fine-tuned model, and a Reinforcement Learning Human Feedback model. The chatbot is designed to provide accurate and helpful responses to user questions.

Link to the chatbot: https://chatbot-for-duke-ai-program-czxuhfx5loyflzempztkpb.streamlit.app/

## Data Sources
1. Web Scraping from Duke websites/Q&A sheet using beautifulsoup
2. Manually collected relevant Q&A
3. Intel/orca_dpo_pairs - for RLHF(DPO)

## System Architecture 
User input -> RAG -> If RAG context has a cosine similarity over 0.5 with the user input, then use user input + RAG context as prompt; otherwise, use user input as prompt -> Fine Tuned Model -> RLHF -> Output

## Modeling Deision

### RAG
Implementation scripts and files are in `./RAG`

Scraped data from relevant websites (ai.meng.duke.edu & https://sites.duke.edu/aipi/new-student-resources/) and stored in weaviate online cluster.



### Fine Tune
Implementation scripts and files are in `./notebook/mistral_finetune*.ipynb`
#### Initial Model: 

Start with a large pre-trained model Mistral-7B-v1.0


#### Integrate QA dataset: 

We define a formatting function to format prompts like {### Instruction:\nUse the provided question to create an instruction that could have been used to generate the answer with an LLM.### Question: {example['question']}\n ### <ans> Answer: {example['answer']}</ans>}


#### Set up the tokenizer: 

Reformat the prompt and tokenize each sample in training set and eval set.


#### Set Low-Rank Matrix Parameter: 

Apply some preprocessing to the model to prepare it for training. For that use the prepare_model_for_kbit_training method from PEFT. Then we apply QLoRA to all the linear layers of the model. Those layers are q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, and lm_head.


#### Start Fine-tuning: 

During the fine-tuning process, only the parameters in the low-rank matrices are updated. The rest of the model's parameters are kept fixed. We firstly finetune the model on the GAIR QA dataset, then we finetune the model on our AIPI QA dataset again based on the first fine tuned model. We use hugging face’s transformers.Trainer to train the model.

### RLHF
Implementation main script location `./RLHF_DPO/DPO_fine_tune_mistral_7b.py`

For efficient training, we use Direct Prefernce Optimizaiton (DPO) to train the model. DPO does not require the creation of a separate reward model or multiple rounds of training, making it easier to implement and train.

## Performance Evaluation
Answer Relevancy, Toxicity, Human Evaluation

## Cost Estimation
1. OpenAI service for evaluation: $3

2. Hugging face Inference Endpoint (Hosted on AWS): $0.5 per hour, 15 mins scale to zero strategy

3. Colab Compute Units: T4:  1.91 per hour approximately, V100: 4.82 per hour approximately

4. Weaviate Vector Database: Free

## Authors
Haoyang Ma, Yibo Liu, Shuaiming Jing, Zihan “Zach” Xing



