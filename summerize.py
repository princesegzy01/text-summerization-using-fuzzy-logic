from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re
import string
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from pprint import pprint
import numpy as np
import skfuzzy as fuzzy
import operator

import numpy as np
import skfuzzy.control as ctrl
# import skfuzzy as fuzz
import sys
import autoCustomRules as acr


stopwords_list = nltk.corpus.stopwords.words('english')

def build_sentence_vector(sentence1, sentence2):
    iterable1 = sentence1.split()
    iterable2 = sentence2.split()


    counter1 = Counter(iterable1)
    counter2 = Counter(iterable2)
    all_items = set(counter1.keys()).union(set(counter2.keys()))
    vector1 = [counter1[k] for k in all_items]
    vector2 = [counter2[k] for k in all_items]
    return vector1, vector2


def cosineSimilarity(v1, v2):
    dot_product = sum(n1 * n2 for n1, n2 in zip(v1, v2) )
    magnitude1 = math.sqrt(sum(n ** 2 for n in v1))
    magnitude2 = math.sqrt(sum(n ** 2 for n in v2))
    return dot_product / (magnitude1 * magnitude2)

def title_word_feature(title, document):

    title_word_feature_values = []

    title_token = word_tokenize(title)

    for sentence in document :
        document_token = word_tokenize(sentence)

        res = (set(document_token).intersection(title_token))
        total = len(res)/ len(title_token)

        title_word_feature_values.append(total)


    return title_word_feature_values


def sentence_length_feature(sentences):

    sentence_legth_feature_result = []
    # get longest length of the sentence

    longest_sentence = 0

    for sentence in sentences :

        sentence_tokenize = word_tokenize(sentence)

        if len(sentence_tokenize) > longest_sentence :
            longest_sentence = len(sentence_tokenize) 
    
    # normalize each length to a list
    for sentence_ in sentences :
        sentence_tokenize = word_tokenize(sentence_)

        sentence_legth_feature_result.append(len(sentence_tokenize) / longest_sentence )
    
    return sentence_legth_feature_result


def get_sentence_location_feature(sentences):

    sentence_location_result = []
    
    for index, sentence in enumerate(sentences) :

        sentence_pos = index + 1
        sentence_location_result.append(1 / sentence_pos)
    
    return sentence_location_result


def get_sentence_numerical_data(sentences):

    numerical_feature = []

    for sentence in sentences :

        sentence_tokenize = word_tokenize(sentence)

        result = [float(s) for s in re.findall(r'-?\d+\.?\d*', sentence)]
        result = len(result)
        sentence_length = len(sentence_tokenize)


        numerical_feature.append(result/sentence_length)
    
    return numerical_feature


def proper_noun_feature(sentences):

    proper_noun_result = []
    
    # loop through senctence in each sentences
    for sentence in sentences:

        sentence_tokenize = word_tokenize(sentence)

        pos_title = nltk.pos_tag(sentence_tokenize)

        nos_nnp_in_sentence = 0


        for k in pos_title:


            if k[1] == "NNP":
                nos_nnp_in_sentence = nos_nnp_in_sentence + 1


        result = nos_nnp_in_sentence / len(sentence_tokenize)

        proper_noun_result.append(result) 
    
    return proper_noun_result


def thematic_keyword(title, sentences):

    document_thematic_result = []

    keyword_dict = dict()

    # loop through each token in the title
    for token in title:

        # convert token to lower case
        token = token.lower()
        
        # make sure token dosent exist in pucntuation or stop word list
        if (token not in list(string.punctuation) and (token not in stopwords_list)):
            
            if token not in keyword_dict :
                keyword_dict[token] = 1
            elif token in keyword_dict :
                keyword_dict[token] = keyword_dict[token] + 1


    # loop through senctence in each sentences
    for sentence in sentences:

        sentence_tokenize = word_tokenize(sentence)

        # loop through token in each sentence
        for token in sentence_tokenize:
            token = token.lower()
            
            # make sure token dosent exist in pucntuation or stop word list
            if (token not in list(string.punctuation) and (token not in stopwords_list)):
            
                if token not in keyword_dict :
                    keyword_dict[token] = 1
                elif token in keyword_dict :
                    keyword_dict[token] = keyword_dict[token] + 1

    from operator import itemgetter
    result = sorted(keyword_dict.items(), key=itemgetter(1))

    #thematic words
    thematic_word = []

    # max_keyword_value = res_tuple[1]

    res_tuple = result[-10:]
    
    #print(res_tuple)

    # get all the nth thematic words from tuple
    for tpl in res_tuple:
        token_name = tpl[0]
        thematic_word.append(token_name)


    # get totla number of thematic words in sentense
    total_sentense_thematic = 0

    for sentence in sentences :
        sentense_tokenize = word_tokenize(sentence)
        result = set(sentense_tokenize).intersection(thematic_word)

        total_sentense_thematic = total_sentense_thematic + len(result)


    # get thematic result for each sentence
    for sentence in sentences :
        sentense_tokenize = word_tokenize(sentence)
        result = set(sentense_tokenize).intersection(thematic_word)

        result = (len(result) / total_sentense_thematic)


        document_thematic_result.append(result)
    return document_thematic_result


def sentence_to_sentence_similarity(sentences):


    all_similarity = []
    sentence_similarity = []


    # loop through senctence in each sentences
    for index, sentence in enumerate(sentences):

        sum_similartity = 0

        for i, sentence_ in enumerate(sentences):

            if index == i :
                continue

            vector1, vector2 = build_sentence_vector(sentence, sentence_)
            similarity = cosineSimilarity(vector1, vector2)

            sum_similartity = sum_similartity + similarity

        all_similarity.append(sum_similartity)
    
  
    maxSimilarity = max(all_similarity)
   
    for similairy in all_similarity :
        sentence_similarity.append(similairy / maxSimilarity)

    return sentence_similarity


def termWeight(sentences):
    
    all_tf_isf_score = []
    sentences_weight = []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_length = len(sentences)
    
    for i in range(sentence_length):
        sentence_weight = tfidf_matrix[i].data
        sum_sentence_weight = sum(sentence_weight)
        all_tf_isf_score.append(sum_sentence_weight)

    max_sentence_weight = max(all_tf_isf_score)


    for sent_weight in all_tf_isf_score:
        sentences_weight.append(sent_weight/max_sentence_weight)

    return sentences_weight


def feature_importantility(count, sentence_feature_list):

    # print(title_word)

    # print(sentence_fuzzy_result , " -- ", sentence_length)

    count = count + 1
    sent_res = []

    for sent_feature in sentence_feature_list:
        sentence_fuzzy_result = single_feature_category(sent_feature)
        maxKey = max(sentence_fuzzy_result.items(), key=operator.itemgetter(1))[0]
        sent_res.append(maxKey)

    return {"S " + str(count): sent_res}
    

def single_feature_category(features):
    cat_low = fuzzy.interp_membership(features_universe, universe_low, features)
    cat_medium = fuzzy.interp_membership(features_universe, universe_medium, features)
    cat_high = fuzzy.interp_membership(features_universe, universe_high, features)
    return dict(low=cat_low, medium=cat_medium, high=cat_high)


def all_feature_category(features_object):

    result = []

    # Add title to array of object
    cat_low = fuzzy.interp_membership(features_universe, universe_low, features_object['title_word'])
    cat_medium = fuzzy.interp_membership(features_universe, universe_medium, features_object['title_word'])
    cat_high = fuzzy.interp_membership(features_universe, universe_high, features_object['title_word'])
    result.append({"title_word" : dict(low=cat_low, medium=cat_medium, high=cat_high)})

     # Add sentence_length to array of object
    cat_low = fuzzy.interp_membership(features_universe, universe_low, features_object['sentence_length'])
    cat_medium = fuzzy.interp_membership(features_universe, universe_medium, features_object['sentence_length'])
    cat_high = fuzzy.interp_membership(features_universe, universe_high, features_object['sentence_length'])
    result.append({"sentence_length" : dict(low=cat_low, medium=cat_medium, high=cat_high)})

     # Add sentence_location to array of object
    cat_low = fuzzy.interp_membership(features_universe, universe_low, features_object['sentence_location'])
    cat_medium = fuzzy.interp_membership(features_universe, universe_medium, features_object['sentence_location'])
    cat_high = fuzzy.interp_membership(features_universe, universe_high, features_object['sentence_location'])
    result.append({"sentence_location" : dict(low=cat_low, medium=cat_medium, high=cat_high)})

     # Add numerical_data to array of object
    cat_low = fuzzy.interp_membership(features_universe, universe_low, features_object['numerical_data'])
    cat_medium = fuzzy.interp_membership(features_universe, universe_medium, features_object['numerical_data'])
    cat_high = fuzzy.interp_membership(features_universe, universe_high, features_object['numerical_data'])
    result.append({"numerical_data" : dict(low=cat_low, medium=cat_medium, high=cat_high)})

     # Add thematic_keyword to array of object
    cat_low = fuzzy.interp_membership(features_universe, universe_low, features_object['thematic_keyword'])
    cat_medium = fuzzy.interp_membership(features_universe, universe_medium, features_object['thematic_keyword'])
    cat_high = fuzzy.interp_membership(features_universe, universe_high, features_object['thematic_keyword'])
    result.append({"thematic_keyword" : dict(low=cat_low, medium=cat_medium, high=cat_high)})

     # Add proper_noun to array of object
    cat_low = fuzzy.interp_membership(features_universe, universe_low, features_object['proper_noun'])
    cat_medium = fuzzy.interp_membership(features_universe, universe_medium, features_object['proper_noun'])
    cat_high = fuzzy.interp_membership(features_universe, universe_high, features_object['proper_noun'])
    result.append({"proper_noun" : dict(low=cat_low, medium=cat_medium, high=cat_high)})

     # Add sentence_similarity to array of object
    cat_low = fuzzy.interp_membership(features_universe, universe_low, features_object['sentence_similarity'])
    cat_medium = fuzzy.interp_membership(features_universe, universe_medium, features_object['sentence_similarity'])
    cat_high = fuzzy.interp_membership(features_universe, universe_high, features_object['sentence_similarity'])
    result.append({"sentence_similarity" : dict(low=cat_low, medium=cat_medium, high=cat_high)})

     # Add title to array of object
    cat_low = fuzzy.interp_membership(features_universe, universe_low, features_object['term_weight'])
    cat_medium = fuzzy.interp_membership(features_universe, universe_medium, features_object['term_weight'])
    cat_high = fuzzy.interp_membership(features_universe, universe_high, features_object['term_weight'])
    result.append({"term_weight" : dict(low=cat_low, medium=cat_medium, high=cat_high)})

    return result
    

def rules_definition(sentence_fuzzy_objects):

    for sentence_fuzzy in sentence_fuzzy_objects : 
        # print(sentence_fuzzy[0]['title_word'])
        
        title_word = sentence_fuzzy[0]['title_word']
        sentence_length = sentence_fuzzy[1]['sentence_length']
        sentence_location = sentence_fuzzy[2]['sentence_location']
        numerical_data = sentence_fuzzy[3]['numerical_data']
        thematic_keyword = sentence_fuzzy[4]['thematic_keyword']
        proper_noun = sentence_fuzzy[5]['proper_noun']
        sentence_similarity = sentence_fuzzy[6]['sentence_similarity']
        term_weight = sentence_fuzzy[7]['term_weight']


        

        # 1. If (title is High) and (Length is High) and (Term is High) and (Position is High) and (Similarity is not High) and (Noun is High) and (Thematic is High) and (Numerical is High) then (Sentence is Important) (1)
        # 2. If (title is High) and (Length is High) and (Term is High) and (Position is High) and (Similarity is High) and (Noun is not High) and (Thematic is High) and (Numerical is HIgh) then (Sentence is Important) (1)
        # 3. If (title is High) and (Length is High) and (Term is High) and (Position is High) and (Similarity is High) and (Noun is High) and (Thematic is not High) and (Numerical is HIgh) then (Sentence is Important) (1)
        # 4. If (title is High) and (Length is not High) and (Term is not High) and (Position is not High) and (Similarity is not High) and (Noun is not High) and (Thematic is not High) and (Numerical is not HIgh) then (Sentence is Unimportant) (1)
        # 5. If (title is not High) and (Length is High) and (Term is not High) and (Position is not High) and (Similarity is not High) and (Noun is not High) and (Thematic is not High) and (Numerical is not HIgh) then (Sentence is Unimportant) (1)
        # 6. If (title is not High) and (Length is not High) and (Term is High) and (Position is not High) and (Similarity is not High) and (Noun is not High) and (Thematic is not High) and (Numerical is not HIgh) then (Sentence is Unimportant) (1)
        # 7. If (title is High) and (Length is not High) and (Term is High) and (Position is High) and (Similarity is not High) and (Noun is not High) and (Thematic is not High) and (Numerical is not HIgh) then (Sentence is Average) (1)
        # 9. If (title is High) and (Length is High) and (Term is not High) and (Position is High) and (similarity is not High) and (Noun is not High) and (Thematic is not High) and (Numerical is not High) then (Sentence is Average) (1)

        rule1 = np.fmax(title_word['high'], sentence_length['high'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        rule2 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['low'], thematic_keyword['high'], proper_noun['low'], sentence_similarity['low'], term_weight['high'])
        rule3 = np.fmax(title_word['low'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['low'], proper_noun['high'], sentence_similarity['low'], term_weight['low'])
        # rule4 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule5 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule6 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule7 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule8 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule9 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule10 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule11 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule12 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule13 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule14 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule15 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule16 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule17 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule18 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule19 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])
        # rule20 = np.fmax(title_word['high'], sentence_length['low'], sentence_location['high'], numerical_data['high'], thematic_keyword['high'], proper_noun['high'], sentence_similarity['high'], term_weight['high'])

        # # Determine the weight and aggregate
        # rule1 = np.fmax(temp_in['hot'], hum_in['low'])
        # rule2 = temp_in['warm']
        # rule3 = np.fmax(temp_in['warm'], hum_in['high'])


        
        
        



        imp1 = np.fmin(rule1, important_sentence)
        imp2 = np.fmin(rule2, average_sentence)
        imp3 = np.fmin(rule3, umimportant_sentence)

        aggregate_membership = np.fmax(imp1, imp2, imp3)

        # # Defuzzify
        result_sentence = fuzzy.defuzz(output_universe, aggregate_membership, 'centroid')

        print(result_sentence)




def fuzzy_rules(sentence_feature_object):
    
    # print(sentence_feature_object)

    universe_features = np.arange(0,1,0.001)
    universe_result = np.arange(0,1,0.01)

    title_word = ctrl.Antecedent(universe_features, 'title_word')
    sentence_length = ctrl.Antecedent(universe_features, 'sentence_length')
    sentence_location = ctrl.Antecedent(universe_features, 'sentence_location')
    numerical_data = ctrl.Antecedent(universe_features, 'numerical_data')
    # thematic_keyword = ctrl.Antecedent(universe_features, 'thematic_keyword')
    # proper_noun = ctrl.Antecedent(universe_features, 'proper_noun')
    # sentence_similarity = ctrl.Antecedent(universe_features, 'sentence_similarity')
    # term_weight = ctrl.Antecedent(universe_features, 'term_weight')


    result = ctrl.Consequent(universe_result, 'result')


    # Auto-membership function population is possible with .automf(3, 5, or 7)
    title_word.automf(3)
    sentence_length.automf(3)
    sentence_location.automf(3)
    numerical_data.automf(3)
    # thematic_keyword.automf(3)
    # proper_noun.automf(3)
    # sentence_similarity.automf(3)
    # term_weight.automf(3)

    # Custom membership functions can be built interactively with a familiar,
    # Pythonic API
    result['low'] = fuzzy.trimf(result.universe, [0.000,0.30,0.500])
    result['medium'] = fuzzy.trimf(result.universe, [0.300,0.500,0.700])
    result['high'] = fuzzy.trimf(result.universe, [0.500,1.0,1.0])

   
    # 'poor'; 'average', or 'good'
   
    rule1  = ctrl.Rule(title_word["poor"] & sentence_length["poor"] & sentence_location["poor"] & numerical_data["poor"], result['low'])
    rule2  = ctrl.Rule(title_word["poor"] & sentence_length["poor"] & sentence_location["poor"] & numerical_data["average"], result['low'])
    rule3  = ctrl.Rule(title_word["poor"] & sentence_length["poor"] & sentence_location["poor"] & numerical_data["good"], result['low'])
    rule4  = ctrl.Rule(title_word["poor"] & sentence_length["poor"] & sentence_location["average"] & numerical_data["poor"], result['low'])
    rule5  = ctrl.Rule(title_word["poor"] & sentence_length["poor"] & sentence_location["average"] & numerical_data["average"], result['low'])
    rule6  = ctrl.Rule(title_word["poor"] & sentence_length["poor"] & sentence_location["average"] & numerical_data["good"], result['low'])
    rule7  = ctrl.Rule(title_word["poor"] & sentence_length["poor"] & sentence_location["good"] & numerical_data["poor"], result['low'])
    rule8  = ctrl.Rule(title_word["poor"] & sentence_length["poor"] & sentence_location["good"] & numerical_data["average"], result['low'])
    rule9  = ctrl.Rule(title_word["poor"] & sentence_length["poor"] & sentence_location["good"] & numerical_data["good"], result['low'])
    rule10  = ctrl.Rule(title_word["poor"] & sentence_length["average"] & sentence_location["poor"] & numerical_data["poor"], result['low'])
    rule11  = ctrl.Rule(title_word["poor"] & sentence_length["average"] & sentence_location["poor"] & numerical_data["average"], result['low'])
    rule12  = ctrl.Rule(title_word["poor"] & sentence_length["average"] & sentence_location["poor"] & numerical_data["good"], result['low'])
    rule13  = ctrl.Rule(title_word["poor"] & sentence_length["average"] & sentence_location["average"] & numerical_data["poor"], result['low'])
    rule14  = ctrl.Rule(title_word["poor"] & sentence_length["average"] & sentence_location["average"] & numerical_data["average"], result['medium'])
    rule15  = ctrl.Rule(title_word["poor"] & sentence_length["average"] & sentence_location["average"] & numerical_data["good"], result['medium'])
    rule16  = ctrl.Rule(title_word["poor"] & sentence_length["average"] & sentence_location["good"] & numerical_data["poor"], result['low'])
    rule17  = ctrl.Rule(title_word["poor"] & sentence_length["average"] & sentence_location["good"] & numerical_data["average"], result['medium'])
    rule18  = ctrl.Rule(title_word["poor"] & sentence_length["average"] & sentence_location["good"] & numerical_data["good"], result['high'])
    rule19  = ctrl.Rule(title_word["poor"] & sentence_length["good"] & sentence_location["poor"] & numerical_data["poor"], result['low'])
    rule20  = ctrl.Rule(title_word["poor"] & sentence_length["good"] & sentence_location["poor"] & numerical_data["average"], result['low'])
    rule21  = ctrl.Rule(title_word["poor"] & sentence_length["good"] & sentence_location["poor"] & numerical_data["good"], result['low'])
    rule22  = ctrl.Rule(title_word["poor"] & sentence_length["good"] & sentence_location["average"] & numerical_data["poor"], result['low'])
    rule23  = ctrl.Rule(title_word["poor"] & sentence_length["good"] & sentence_location["average"] & numerical_data["average"], result['medium'])
    rule24  = ctrl.Rule(title_word["poor"] & sentence_length["good"] & sentence_location["average"] & numerical_data["good"], result['high'])
    rule25  = ctrl.Rule(title_word["poor"] & sentence_length["good"] & sentence_location["good"] & numerical_data["poor"], result['low'])
    rule26  = ctrl.Rule(title_word["poor"] & sentence_length["good"] & sentence_location["good"] & numerical_data["average"], result['high'])
    rule27  = ctrl.Rule(title_word["poor"] & sentence_length["good"] & sentence_location["good"] & numerical_data["good"], result['high'])
    rule28  = ctrl.Rule(title_word["average"] & sentence_length["poor"] & sentence_location["poor"] & numerical_data["poor"], result['low'])
    rule29  = ctrl.Rule(title_word["average"] & sentence_length["poor"] & sentence_location["poor"] & numerical_data["average"], result['low'])
    rule30  = ctrl.Rule(title_word["average"] & sentence_length["poor"] & sentence_location["poor"] & numerical_data["good"], result['low'])
    rule31  = ctrl.Rule(title_word["average"] & sentence_length["poor"] & sentence_location["average"] & numerical_data["poor"], result['low'])
    rule32  = ctrl.Rule(title_word["average"] & sentence_length["poor"] & sentence_location["average"] & numerical_data["average"], result['medium'])
    rule33  = ctrl.Rule(title_word["average"] & sentence_length["poor"] & sentence_location["average"] & numerical_data["good"], result['medium'])
    rule34  = ctrl.Rule(title_word["average"] & sentence_length["poor"] & sentence_location["good"] & numerical_data["poor"], result['low'])
    rule35  = ctrl.Rule(title_word["average"] & sentence_length["poor"] & sentence_location["good"] & numerical_data["average"], result['medium'])
    rule36  = ctrl.Rule(title_word["average"] & sentence_length["poor"] & sentence_location["good"] & numerical_data["good"], result['high'])
    rule37  = ctrl.Rule(title_word["average"] & sentence_length["average"] & sentence_location["poor"] & numerical_data["poor"], result['low'])
    rule38  = ctrl.Rule(title_word["average"] & sentence_length["average"] & sentence_location["poor"] & numerical_data["average"], result['medium'])
    rule39  = ctrl.Rule(title_word["average"] & sentence_length["average"] & sentence_location["poor"] & numerical_data["good"], result['medium'])
    rule40  = ctrl.Rule(title_word["average"] & sentence_length["average"] & sentence_location["average"] & numerical_data["poor"], result['medium'])
    rule41  = ctrl.Rule(title_word["average"] & sentence_length["average"] & sentence_location["average"] & numerical_data["average"], result['medium'])
    rule42  = ctrl.Rule(title_word["average"] & sentence_length["average"] & sentence_location["average"] & numerical_data["good"], result['medium'])
    rule43  = ctrl.Rule(title_word["average"] & sentence_length["average"] & sentence_location["good"] & numerical_data["poor"], result['medium'])
    rule44  = ctrl.Rule(title_word["average"] & sentence_length["average"] & sentence_location["good"] & numerical_data["average"], result['medium'])
    rule45  = ctrl.Rule(title_word["average"] & sentence_length["average"] & sentence_location["good"] & numerical_data["good"], result['medium'])
    rule46  = ctrl.Rule(title_word["average"] & sentence_length["good"] & sentence_location["poor"] & numerical_data["poor"], result['low'])
    rule47  = ctrl.Rule(title_word["average"] & sentence_length["good"] & sentence_location["poor"] & numerical_data["average"], result['medium'])
    rule48  = ctrl.Rule(title_word["average"] & sentence_length["good"] & sentence_location["poor"] & numerical_data["good"], result['high'])
    rule49  = ctrl.Rule(title_word["average"] & sentence_length["good"] & sentence_location["average"] & numerical_data["poor"], result['medium'])
    rule50  = ctrl.Rule(title_word["average"] & sentence_length["good"] & sentence_location["average"] & numerical_data["average"], result['medium'])
    rule51  = ctrl.Rule(title_word["average"] & sentence_length["good"] & sentence_location["average"] & numerical_data["good"], result['medium'])
    rule52  = ctrl.Rule(title_word["average"] & sentence_length["good"] & sentence_location["good"] & numerical_data["poor"], result['high'])
    rule53  = ctrl.Rule(title_word["average"] & sentence_length["good"] & sentence_location["good"] & numerical_data["average"], result['medium'])
    rule54  = ctrl.Rule(title_word["average"] & sentence_length["good"] & sentence_location["good"] & numerical_data["good"], result['high'])
    rule55  = ctrl.Rule(title_word["good"] & sentence_length["poor"] & sentence_location["poor"] & numerical_data["poor"], result['low'])
    rule56  = ctrl.Rule(title_word["good"] & sentence_length["poor"] & sentence_location["poor"] & numerical_data["average"], result['low'])
    rule57  = ctrl.Rule(title_word["good"] & sentence_length["poor"] & sentence_location["poor"] & numerical_data["good"], result['low'])
    rule58  = ctrl.Rule(title_word["good"] & sentence_length["poor"] & sentence_location["average"] & numerical_data["poor"], result['low'])
    rule59  = ctrl.Rule(title_word["good"] & sentence_length["poor"] & sentence_location["average"] & numerical_data["average"], result['medium'])
    rule60  = ctrl.Rule(title_word["good"] & sentence_length["poor"] & sentence_location["average"] & numerical_data["good"], result['high'])
    rule61  = ctrl.Rule(title_word["good"] & sentence_length["poor"] & sentence_location["good"] & numerical_data["poor"], result['low'])
    rule62  = ctrl.Rule(title_word["good"] & sentence_length["poor"] & sentence_location["good"] & numerical_data["average"], result['high'])
    rule63  = ctrl.Rule(title_word["good"] & sentence_length["poor"] & sentence_location["good"] & numerical_data["good"], result['high'])
    rule64  = ctrl.Rule(title_word["good"] & sentence_length["average"] & sentence_location["poor"] & numerical_data["poor"], result['low'])
    rule65  = ctrl.Rule(title_word["good"] & sentence_length["average"] & sentence_location["poor"] & numerical_data["average"], result['medium'])
    rule66  = ctrl.Rule(title_word["good"] & sentence_length["average"] & sentence_location["poor"] & numerical_data["good"], result['high'])
    rule67  = ctrl.Rule(title_word["good"] & sentence_length["average"] & sentence_location["average"] & numerical_data["poor"], result['medium'])
    rule68  = ctrl.Rule(title_word["good"] & sentence_length["average"] & sentence_location["average"] & numerical_data["average"], result['medium'])
    rule69  = ctrl.Rule(title_word["good"] & sentence_length["average"] & sentence_location["average"] & numerical_data["good"], result['medium'])
    rule70  = ctrl.Rule(title_word["good"] & sentence_length["average"] & sentence_location["good"] & numerical_data["poor"], result['high'])
    rule71  = ctrl.Rule(title_word["good"] & sentence_length["average"] & sentence_location["good"] & numerical_data["average"], result['medium'])
    rule72  = ctrl.Rule(title_word["good"] & sentence_length["average"] & sentence_location["good"] & numerical_data["good"], result['high'])
    rule73  = ctrl.Rule(title_word["good"] & sentence_length["good"] & sentence_location["poor"] & numerical_data["poor"], result['low'])
    rule74  = ctrl.Rule(title_word["good"] & sentence_length["good"] & sentence_location["poor"] & numerical_data["average"], result['high'])
    rule75  = ctrl.Rule(title_word["good"] & sentence_length["good"] & sentence_location["poor"] & numerical_data["good"], result['high'])
    rule76  = ctrl.Rule(title_word["good"] & sentence_length["good"] & sentence_location["average"] & numerical_data["poor"], result['high'])
    rule77  = ctrl.Rule(title_word["good"] & sentence_length["good"] & sentence_location["average"] & numerical_data["average"], result['medium'])
    rule78  = ctrl.Rule(title_word["good"] & sentence_length["good"] & sentence_location["average"] & numerical_data["good"], result['high'])
    rule79  = ctrl.Rule(title_word["good"] & sentence_length["good"] & sentence_location["good"] & numerical_data["poor"], result['high'])
    rule80  = ctrl.Rule(title_word["good"] & sentence_length["good"] & sentence_location["good"] & numerical_data["average"], result['high'])
    rule81  = ctrl.Rule(title_word["good"] & sentence_length["good"] & sentence_location["good"] & numerical_data["good"], result['high'])
    
    tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46, rule47, rule48, rule49, rule50, rule51, rule52, rule53, rule54, rule55, rule56, rule57, rule58, rule59, rule60, rule61, rule62, rule63, rule64, rule65, rule66, rule67, rule68, rule69, rule70, rule71, rule72, rule73, rule74, rule75, rule76, rule77, rule78, rule79, rule80, rule81])
    # tipping_ctrl = acr.customRule()
    #tipping control simmulation
    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
    # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
    tipping.input['title_word'] = sentence_feature_object['title_word']
    tipping.input['sentence_length'] = sentence_feature_object['sentence_length']
    tipping.input['sentence_location'] = sentence_feature_object['sentence_location']
    tipping.input['numerical_data'] = sentence_feature_object['numerical_data']
    # tipping.input['thematic_keyword'] = sentence_feature_object['thematic_keyword']
    # tipping.input['proper_noun'] = sentence_feature_object['proper_noun']
    # tipping.input['sentence_similarity'] = sentence_feature_object['sentence_similarity']
    # tipping.input['term_weight'] = sentence_feature_object['term_weight']

   
    # tipping.inputs(sentence_feature_object)

    # Crunch the numbers
    tipping.compute()

    return tipping.output['result']
def starter ():
    raw_document_file = open("doc.txt","r")

    document_content = raw_document_file.read()

    text = document_content.split('\n', 1)

    sentences_list = []
    Words = dict()

    title = text[0]
    text = text[1].replace(u"\u2018", '\'').replace(u"\u2019", '\'').replace(u"\u201c",'"').replace(u"\u201d", '"')

    title_text = title

    #preprocess title
    stemmer = nltk.stem.porter.PorterStemmer()

    #just remove stop words

    title_tokens = word_tokenize(title_text)
    title_tokens = [ stemmer.stem(token) for token in title_tokens if token not in stopwords_list]

    title_stemmed = ' '.join(title_tokens)
 
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    detected_sentences = sentence_detector.tokenize(text.strip())

    #loop through the detected sentences 
    # and take each sentence for preprocessing


    #print(detected_sentences)
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    # wordnet_lemmatizer.lemmatize(‘dogs’)

    for detected_sentence in detected_sentences :


        words_tokens = word_tokenize(detected_sentence)
        sentence = [wordnet_lemmatizer.lemmatize(word) for word in words_tokens if word not in stopwords_list]

        sentence = ' '.join(sentence) 

        sentences_list.append(sentence)


    title_word_feature_value = title_word_feature(title_stemmed,sentences_list)
    sentence_length_feature_value = sentence_length_feature(sentences_list)
    sentence_location_feature_value = get_sentence_location_feature(sentences_list)
    numerical_data_feature_value = get_sentence_numerical_data(sentences_list)
    thematic_keyword_feature_value = thematic_keyword(title_stemmed, sentences_list)
    proper_noun_feature_value = proper_noun_feature(sentences_list)
    sentence_similarity_feature_value = sentence_to_sentence_similarity(sentences_list)
    term_weight_feature_value = termWeight(sentences_list)

    sentences_feature_list = []

    fuzzy_list = []

    sentence_total_object_result = []

    indexCounter = 0

    for title_word,sentence_length,sentence_location, numerical_data, thematic_key, proper_noun,sentence_similarity,term_weight in zip(title_word_feature_value, sentence_length_feature_value, sentence_location_feature_value, numerical_data_feature_value, thematic_keyword_feature_value, proper_noun_feature_value,sentence_similarity_feature_value, term_weight_feature_value) : 
    

        sin_feature_obj = {
            'title_word': title_word,
            'sentence_length': sentence_length,
            'sentence_location': sentence_location,
            'numerical_data': numerical_data,
            'thematic_keyword': thematic_key,
            'proper_noun': proper_noun,
            'sentence_similarity': sentence_similarity,
            'term_weight': term_weight,
        }
        
        result = fuzzy_rules(sin_feature_obj)

        # result_object = {"sn" : indexCounter, "value" : result}
        # fuzzy_list.append(result_object)

        result_object = (indexCounter, result)
        fuzzy_list.append(result_object)
        
        indexCounter = indexCounter + 1
   

    # use 20%  as compression ratio
    total_sentence = len(detected_sentences)
    compressionNumber = total_sentence * 0.20
    compressionNumber = int(compressionNumber)


    #  sort the result according to the fuzzy result
    #  and select up to compression rate
    result = sorted(fuzzy_list, key=lambda t: t[1], reverse=True)[:compressionNumber]
    result2 = sorted(result, key=lambda t: t[0])


    for index in result2:
        print(detected_sentences[index[0]])

    # for i in range(0, compressionNumber-1):
        # print(detected_sentences[fuzzy_list[i]['sn']])


starter()