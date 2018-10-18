# Text summerization using fuzzy logic

This is an implementation of a text summeration from a txt file using fuzzy logic

## Language & Package
* python 3.6
* skfuzzy from sci-kit fuzzy
* nltk



### preprocessing

* Lemmatization
* Stopwords
* sentence tokenizing

## Fuzzifier
getting features from the extracted sentence

### Features Extraction

* Tight weight
* Sentence location
* Sentence length
* Thematic word
* Term weight
* Sentence similarity
* Proper noun
* Numerical data

### Membership function

* Triangular membership function
* Auto membership function

### Inference Engine
* Rule based

## Defuzifier
Converting the fuzzified values to crisp for sentence selection


## Senetence Selection
* compression rate is 20%, which means.. get top 20% sentence with highest score from the list of all the sentence in the corpus.