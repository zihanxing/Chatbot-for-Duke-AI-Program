# Chatbot for Duke AI MEng Program

## Data Sources
1. Web Scraping from Duke websites/Q&A sheet using beautifulsoup
2. Manually collected relevant Q&A

## System Architecture 
User input -> RAG -> If RAG context has a cosine similarity over 0.5 with the user input, then use user input + RAG context as prompt; otherwise, use user input as prompt -> Fine Tuned Model -> RLHF -> Output

## Modeling Deision
### RAG
Scraped data from relevant websites (ai.meng.duke.edu & https://sites.duke.edu/aipi/new-student-resources/) and stored in weaviate online cluster.

### Fine Tune

## Performance Evaluation
Answer Relevancy, Toxicity, Human Evaluation

## Cost Estimation


## RLHF

