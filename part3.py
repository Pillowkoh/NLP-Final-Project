import numpy as np
from part2 import *

'''
(i) forward_algorithm(x, f, possible_states)
Implementation of forward algorithm to calculate score for a given input sequence x

Parameters:
x: List of word tokens from sentence                             Type: list[str]
f: Dictionary of feature weights                                 Type: Dict{features: weights}
possible_states: List of possible tags                           Type: list[str]
'''
def forward_algorithm(x, f, possible_states):
    forward_scores = np.zeros((len(x), len(possible_states)))
    threshold = 700

    for i in range(len(possible_states)):
        t_key = "transition: START+"+possible_states[i]
        t_prob = f.get(t_key, -2**31)
        e_key = "emission: "+possible_states[i]+"+"+x[0]
        e_prob = f.get(e_key, -2**31)
        forward_scores[0, i] = t_prob + e_prob
    
    for i in range(1, len(x)):
        for k in range(len(possible_states)):
            temp_score = 0
            for j in range(len(possible_states)):
                t_key = "transition: "+possible_states[j]+"+"+possible_states[k]
                t_prob = f.get(t_key, -2**31)
                e_key = "emission: "+possible_states[k]+"+"+x[i]
                e_prob = f.get(e_key, -2**31)
                score = e_prob + t_prob + forward_scores[i-1,j]
                if score > threshold:
                    score = threshold
                temp_score += np.exp(score)
                
            forward_scores[i, k] = np.log(temp_score) if temp_score else -2**31

    forward_prob = 0
    for j in range(len(possible_states)):
        t_key = "transition: "+possible_states[j]+"+STOP"
        t_prob = f.get(t_key, -2**31)
        score = t_prob + forward_scores[len(x)-1,j]
        if score > threshold:
            score = threshold
        overall_score = np.exp(score)
        forward_prob += overall_score
    if forward_prob > 0:
        alpha = np.log(forward_prob)
    else:
        alpha = threshold
        
    return forward_scores, alpha

'''
(i) CRF_loss(train_dataset, f, possible_states)
Implementation of CRF loss function to calculate loss for a given input dataset

Parameters:
train_dataset: List of sentences from training data                 Type: list[list[(str, str)]]
f: Dictionary of feature weights                                    Type: Dict{features: weights}
possible_states: List of possible tags                              Type: list[str]
'''
def CRF_loss(train_dataset,f,possible_states):
    temp_loss = 0
    for sentence in train_dataset:
        x = [token_tag_pair[0] for token_tag_pair in sentence]
        y = [token_tag_pair[1] for token_tag_pair in sentence]
        _, alpha = forward_algorithm(x, f, possible_states)
        temp_loss += calculate_score(x,y,f) - alpha

    crf_loss = -temp_loss
    return crf_loss

'''
    (ii) backward_algorithm(x, f, possible_states)
    Implementation of backward algorithm to calculate score for a given input sequence x

    Parameters:
    x: List of word tokens from sentence                            Type: list[str]
    f: Dictionary of feature weights                                Type: Dict{features: weights}
    possible_states: List of possible tags                          Type: list[str]
'''
def backward_algorithm(x, f, possible_states):
    
    backward_scores = np.zeros((len(x), len(possible_states)))
    threshold = 700
    
    for i in range(len(possible_states)):
        t_key = "transition: "+ possible_states[i] +"+STOP"
        t_prob = f.get(t_key, -2**31)
        backward_scores[len(x)-1, i] = t_prob

    for i in range(len(x)-1,0,-1):
        for j  in range(len(possible_states)):
            temp_score = 0 
            for k in range(len(possible_states)):
                t_key = "transition: " + possible_states[j] + "+" + possible_states[k]
                e_key = "emission: "+ possible_states[k] + "+" + x[i] 
                t_prob = f.get(t_key, -2**31)
                e_prob = f.get(e_key, -2**31)
                temp_score += np.exp(min(e_prob + t_prob + backward_scores[i, k], threshold))
            
            if temp_score!=0:
                backward_scores[i-1, j] = np.log(temp_score)
            else:
                backward_scores[i-1, j] = -2**31
    
    backward_prob = 0
    for i in range(len(possible_states)):
        t_key = "transition: " + "START" + "+" + possible_states[i]
        e_key = "emission: " + possible_states[i] + "+" + x[0]
        t_prob = f.get(t_key, -2**31)
        e_prob = f.get(e_key, -2**31)
        overall_score = np.exp(min(e_prob + t_prob + backward_scores[0, i], threshold))
        backward_prob += overall_score
        
    if backward_prob!=0:
        beta = np.log(backward_prob)
    else:
        beta = -threshold    

    return backward_scores, beta

'''
    (ii) forward_backward_algorithm(x, f, possible_states)
    Full implementation of forward-backward algorithm

    Parameters:
    x: List of word tokens from sentence                            Type: list[str]
    f: Dictionary of feature weights                                Type: Dict{features: weights}
    possible_states: List of possible tags                          Type: list[str]
'''
def forward_backward(x, f, possible_states):
    
    threshold = 700
    
    forward_scores, alpha = forward_algorithm(x, f, possible_states)
    forward_prob = np.exp(min(alpha, threshold))
    backward_scores, beta = backward_algorithm(x, f, possible_states)
    backward_prob = np.exp(min(beta, threshold))
    feature_expected_counts = {}

    for i in range(len(x)):
        for j in range(len(possible_states)):
            e_key = "emission: " + possible_states[j] + "+" + x[i]
            feature_expected_counts[e_key] = feature_expected_counts.get(e_key, 0.0) + np.exp(min(forward_scores[i, j] + backward_scores[i, j] - alpha, threshold))
    
    for i in range(len(possible_states)):
        start_t_key =  "transition: " + "START" + "+" + possible_states[i]
        feature_expected_counts[start_t_key] = feature_expected_counts.get(start_t_key, 0.0) + np.exp(min(forward_scores[0, i] + backward_scores[0, i] - alpha, threshold))
        stop_t_key =  "transition: " + possible_states[i] + "+"  + "STOP"
        feature_expected_counts[stop_t_key] = feature_expected_counts.get(stop_t_key, 0.0) + np.exp(min(forward_scores[len(x)-1, i] + backward_scores[len(x)-1, i] - alpha, threshold))
    
    for i in range(len(possible_states)):
        for j in range(len(possible_states)):
            t_key =  "transition: " + possible_states[i] + "+"  + possible_states[j]
            t_prob = f.get(t_key, -2**31) 
            total = 0
            for k in range(len(x)-1):
                e_key =  "emission: " + possible_states[j] + "+"  + x[k+1]
                e_prob = f.get(e_key, -2**31)

                total += np.exp(min(forward_scores[k, i] + backward_scores[k+1, j] + t_prob + e_prob - alpha, threshold))

            feature_expected_counts[t_key] = total
    
    return feature_expected_counts

'''
    (ii) get_feature_count(x, y, feature_dict)
    Calculate emission and transition feature counts for a given input sequence x and tag sequence y

    Parameters:
    x: List of word tokens from sentence                            Type: list[str]
    y: List of tags for each word in x                              Type: list[str]
'''
def get_feature_count(x, y):
    n = len(x)
    feature_count = {}
    
    for i in range(n):
        formatted_word = x[i]
        emission_key = "emission: "+ y[i] + "+" + formatted_word
        feature_count[emission_key] = feature_count.get(emission_key, 0) + 1
    
    updated_y = ["START"] + y + ["STOP"]
    for i in range(1, n+2):
        prev_y = updated_y[i-1]
        y_i = updated_y[i]
        transition_key = "transition: " + prev_y + "+" + y_i
        feature_count[transition_key] = feature_count.get(transition_key, 0) + 1
    
    return feature_count
    
'''
    (ii) compute_gradients(train_dataset, f, possible_states)
    Calculate gradients (a gradient vector) based on the forward and backward scores for each feature

    Parameters:
    train_dataset: List of sentences                            Type: list[(list[str], list[str])]
    f: Dictionary of feature weights                            Type: Dict{features: weights}
    possible_states: List of possible tags                      Type: list[str]
'''
def compute_gradients(train_dataset, f, possible_states):
    feature_gradients = {}
    for sentence in train_dataset:
        x = [token_tag_pair[0] for token_tag_pair in sentence]
        y = [token_tag_pair[1] for token_tag_pair in sentence]
        feature_expected_counts = forward_backward(x, f, possible_states)
        actual_counts = get_feature_count(x, y)
        
        for k, v in feature_expected_counts.items():
            feature_gradients[k] = feature_gradients.get(k, 0) + v
            
        for k, v in actual_counts.items():
            feature_gradients[k] = feature_gradients.get(k, 0) - v

    return feature_gradients

if __name__ == '__main__':
    possible_states = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative', 'B-neutral', 'I-neutral']

    print('Reading train file...')
    train_dataset = read_train_file('./dataset/train')

    print('Getting features...')
    transition_count, emission_count, state_count = get_transition_emission_counts(train_dataset)
    f = {}
    add_e_prob(f, emission_count, state_count)
    add_t_prob(f, transition_count, state_count)

    print('Running checks for calculation of gradients...')
    feature_key_checks = ['emission: O+the', 'transition: START+O', 'transition: O+O']
    feature_gradients = compute_gradients(train_dataset, f, possible_states)
    loss1 = CRF_loss(train_dataset, f, possible_states)
    delta = 1e-6

    for feature_key in feature_key_checks:
        print("Running", feature_key)
        new_f = f.copy()
        new_f[feature_key] += delta

        loss2 = CRF_loss(train_dataset, new_f, possible_states)

        numerical_gradient = (loss2 - loss1) / delta
        analytical_gradient = feature_gradients[feature_key]
        print(numerical_gradient, analytical_gradient)
        
        # SANITY CHECK
        assert(abs(numerical_gradient - analytical_gradient) / max(abs(numerical_gradient), 1e-8) <= 1e-3)