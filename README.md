
We propose here a package for our **NLP** course. We designed here multiples classification and generation models to explore a selcted dataset.

To use our package, you can use one of our scripts `train.py`, `evaluation.py` or `generate.py`, with a config.
You can find config examples in the `examples/` directory.

e.g : `python train.py -c config.yml`

# **Dataset Datasheet: TripAdvisor Review Rating**

## **Dataset Summary**
The **TripAdvisor Review Rating** dataset consists of user reviews from TripAdvisor, labeled with star ratings. This dataset is useful for sentiment analysis, text classification, and natural language processing (NLP) tasks, such as training models to predict review ratings based on review content.

## **Dataset Details**
- **Dataset Name**: TripAdvisor Review Rating  
- **Source**: [Hugging Face Dataset Hub](https://huggingface.co/datasets/jniimi/tripadvisor-review-rating)  
- **Provider**: jniimi  
- **Domain**: Online reviews, Travel & Tourism  
- **Language**: English  
- **Number of Records**: 200k rows
- **License**: Apache-2.0  

## **Data Structure**
- **Features**:
  - `text` (string): The review text written by a user.
  - `rating` (integer): The corresponding star rating (1-5).

## **Intended Use**
- Sentiment analysis
- Star rating prediction
- Opinion mining
- NLP model training and evaluation  

## **Potential Applications**
- Training machine learning models for sentiment classification
- Understanding customer satisfaction trends
- Analyzing travel and tourism reviews for businesses

## **Ethical Considerations & Risks**
- **Bias**: The dataset may have biases based on the user demographics, review styles, or cultural factors.
- **Privacy**: Ensure that reviews do not contain personally identifiable information (PII).
- **Misuse**: Models trained on this dataset should be tested to avoid overfitting and unfair ratings.

## **References**
- Dataset hosted on Hugging Face: [TripAdvisor Review Rating](https://huggingface.co/datasets/jniimi/tripadvisor-review-rating)  
- TripAdvisor Website: [https://www.tripadvisor.com](https://www.tripadvisor.com)

## **Observations**

At the end of this project, we made the observation that the dataset was not well balaced and lead to bad metrics that we fixed by rebalancing it with data augmentation.

# **Data Augmentation**

As specified above, the data augmentation was pretty usefull for re balancing the dataset and give a high performance improvments.
The user can select the: `balance_percentage`, `augmentation_methods`, `augmentation_workers` (for parallelism).

## **Available Augmentations**

- synonym
- contextual
- random
- sentence_shuffle
- word_deletion

# **Tokenizer**

# **Vectorizer**

# **Models**

## **Classification**
## **Generative**