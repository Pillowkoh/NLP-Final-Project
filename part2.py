import numpy as np
from conlleval import evaluate, evaluate_conll_file
from part1 import *

'''
(i): calculate_score(x,y)
Helps to calulate the score for a given pair of input and output sequence pair (x,y)
Based on 2 features, emission and transition

Parameters:
x: List of tokens, e.g. x = x1, x2, ..., xn     Type: list[str]
y: List of tokens, e.g. y = y1, y2, ..., yn     Type: list[str]
f: Dictionary of feature weights                Type: Dict{features: weights}
'''

def calculate_score(x,y,f):
    assert len(x) == len(y)

    feature_count = {}

    prev_tag = 'START'
    score = 0

    length = len(x)
    for i in range(length):
        e_key = "emission: " + y[i] + '+' + x[i]
        t_key = "transition: " + prev_tag + '+' + y[i]

        if e_key in f.keys():
            feature_count[e_key] = feature_count.get(e_key, 0) + 1

        if t_key in f.keys():
            feature_count[t_key] = feature_count.get(t_key, 0) + 1

        prev_tag = y[i]
        
    t_key = "transition: " + prev_tag + '+' + 'STOP'
    if t_key in f.keys():
            feature_count[t_key] = feature_count.get(t_key, 0) + 1

    for feature, count in feature_count.items():
        score += f[feature] * count

    return score

'''
(ii) virterbi(sentence, f)
Performs decoding using the Viterbi algorithm to find the most probable output sequence y*
for a given input sequence x*

Parameters:
sentence: List of tokens, e.g. x = x1, x2, ..., xn     Type: list[str]
f: Dictionary of feature weights                   Type: Dict{features: weights}
'''
def viterbi(x, possible_states, f, default_index=0):
    n = len(x)
    d = len(possible_states)
    scores = np.full((n, d), -np.inf)
    bp = np.full((n, d), default_index, dtype=np.int32)

    for i in range(len(possible_states)):
        t_key = "transition: START"+possible_states[i]
        e_key = "emission: "+possible_states[i]+"+"+x[0]
        t_prob = f.get(t_key, -2**31)
        e_prob = f.get(e_key, -2**31)
        scores[0, i] = t_prob + e_prob
    
    for i in range(1, n):
        for k in range(len(possible_states)):
            for j in range(len(possible_states)):
                t_key = "transition: "+possible_states[k]+"+"+possible_states[j]
                e_key = "emission: "+possible_states[j]+"+"+x[i]
                t_prob = f.get(t_key, -2**31)
                e_prob = f.get(e_key, -2**31)
                overall_score = e_prob + t_prob + scores[i-1, k]
                if overall_score > scores[i, j]:
                    scores[i, j] = overall_score
                    bp[i,j] = k
    
    highest_score = -2**31
    highest_bp = default_index
    for i in range(len(possible_states)):
        t_key = "transition: "+possible_states[i]+"+STOP"
        t_prob = f.get(t_key, -2**31)
        overall_score = t_prob + scores[n-1, i]
        
        if overall_score > highest_score:
            highest_score = overall_score
            highest_bp = i
    
    result = [possible_states[highest_bp]]
    prev_bp = highest_bp
    for i in range(n-1, 0, -1):
        prev_bp = bp[i, prev_bp]
        output = possible_states[prev_bp]
        result = [output] + result
    
    return result

'''
(ii) decode_file(path, states, f, output_filename)
Decodes a file of sentences using the Viterbi algorithm

Parameters:
path: Path to the file to be decoded               Type: str
states: List of possible states                    Type: list[str]
f: Dictionary of feature weights                   Type: Dict{features: weights}
output_filename: Name of the output file           Type: str
'''
def decode_file(path, states, f, output_filename):
    default_index = states.index('O')
    sentences = list()

    with open(path) as file:
        lines = file.readlines()
        sentence = list()
        for line in lines:
            formatted_line = line.strip()   
            
            if(len(formatted_line) == 0):
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            sentence.append(formatted_line)

    with open(output_filename, "w") as wf:
        for sentence in sentences:
            pred_sentence = viterbi(sentence, states, f, default_index)        
            for i in range(len(sentence)):
                wf.write(sentence[i] + " " + pred_sentence[i] + "\n")
                
            wf.write("\n")

'''
    get_tags(pred, gold)
    Helper function to read input files to get tags for both gold and prediction

    Parameters:
    pred: Path to the file with predicted tags       Type: str
    gold: Path to the file with gold tags            Type: str
'''
def get_tags(pred,gold):
    f_pred = open(pred,encoding = 'utf-8')
    f_gold = open(gold,encoding = 'utf-8')
    data_pred = f_pred.readlines()
    data_gold = f_gold.readlines()
    gold_tags = list()
    pred_tags = list()
    
    for sentence in range(len(data_gold)):
        words_pred = data_pred[sentence].strip().split(' ')
        words_gold = data_gold[sentence].strip().split(' ')  
        if len(words_gold)==1:
            continue
        # Write original word and predicted tags
        gold_tags.append(words_gold[1])
        pred_tags.append(words_pred[1])
        # End of sentence, write newline
    return gold_tags,pred_tags

if __name__ == '__main__':
    possible_states = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative', 'B-neutral', 'I-neutral']
    
    print('Reading train file...')
    train_dataset = read_train_file('./dataset/train')

    print('Getting features...')
    transition_count, emission_count, state_count = get_transition_emission_counts(train_dataset)
    f = {}
    add_e_prob(f, emission_count, state_count)
    add_t_prob(f, transition_count, state_count)

    print('Decoding dataset/dev.in...')
    decode_file("dataset/dev.in", possible_states, f, 'partial/dev.p2.out')

    print('Ran decoding on dataset/dev.in. Output: partial/dev.p2.out')

    print('Running evaluation using conlleval...')
    g_tags, p_tags = get_tags('partial/dev.p2.out', 'dataset/dev.out')
    print(evaluate(g_tags,p_tags,verbose=True))


