import numpy as np
from conlleval import evaluate, evaluate_conll_file
from part1 import *
from part2 import get_tags

def get_feature_count_p6i(train_dataset):
    # added new bigram: y_i-1 + y_i + x_i-1
    # added new bigram: y_i-1 + y_i + x_i+1

    # added new token transition: y_i-1 + x_i-1 + x_i
    feature_counts = {}
    emission_count = {}
    transition_count = {}
    # token_transition_count = {}
    uni_count = {}
    bi_count = {}
    state_count = {}
    start_state = "START"
    stop_state = "STOP"

    for sentence in train_dataset:
        x = [token_tag_pair[0] for token_tag_pair in sentence]
        y = [token_tag_pair[1] for token_tag_pair in sentence]

        n = len(x)

        # START state
        if uni_count.get(start_state) == None:
            uni_count[start_state] = {}
        uni_count[start_state][x[0]] = uni_count[start_state].get(x[0], 0) + 1

        state_count[start_state] = state_count.get(start_state, 0) + 1

        # STOP state
        if transition_count.get(y[n-1]) == None:
            transition_count[y[n-1]] = {}
        transition_count[y[n-1]][stop_state] = transition_count[y[n-1]].get(stop_state, 0) + 1
        
        if uni_count.get(stop_state) == None:
            uni_count[stop_state] = {}
        uni_count[stop_state][x[n-1]] = uni_count[stop_state].get(x[n-1], 0) + 1

        if bi_count.get((y[n-1],stop_state)) == None:
            bi_count[(y[n-1],stop_state)] = {}
        bi_count[(y[n-1],stop_state)][x[n-1]] = bi_count[(y[n-1],stop_state)].get(x[n-1], 0) +1

        state_count[stop_state] = state_count.get(stop_state, 0) + 1

        for i in range(n):

            # First word
            if i == 0:
                if emission_count.get(y[i]) == None:
                    emission_count[y[i]] = {}
                emission_count[y[i]][x[i]] = emission_count[y[i]].get(x[i], 0) + 1
                        
                if transition_count.get(start_state) == None:
                    transition_count[start_state] = {}
                transition_count[start_state][y[i]] = transition_count[start_state].get(y[i], 0) + 1

                if bi_count.get((start_state, y[i])) == None:
                    bi_count[(start_state, y[i])] = {}
                bi_count[(start_state, y[i])][x[i]] = bi_count[(start_state, y[i])].get(x[i], 0) + 1

                if n>1:
                    if uni_count.get(y[i]) == None:
                        uni_count[y[i]] = {}
                    uni_count[y[i]][x[i+1]] = uni_count[y[i]].get(x[i+1], 0) + 1
                    bi_count[(start_state, y[i])][x[i+1]] = bi_count[(start_state, y[i])].get(x[i+1], 0) + 1 # FORWARD BIGRAM

                state_count[(start_state, y[i])] = state_count.get((start_state, y[i]), 0) + 1                

            # Last word
            elif i == n-1:
                if emission_count.get(y[i]) == None:
                    emission_count[y[i]] = {}
                emission_count[y[i]][x[i]] = emission_count[y[i]].get(x[i], 0) + 1

                if uni_count.get(y[i]) == None:
                    uni_count[y[i]] = {}
                uni_count[y[i]][x[i-1]] = uni_count[y[i]].get(x[i-1], 0) + 1

                if transition_count.get(y[i-1]) == None:
                    transition_count[y[i-1]] = {}
                transition_count[y[i-1]][y[i]] = transition_count[y[i-1]].get(y[i], 0) + 1

                # if token_transition_count.get(y[i-1]) == None:
                #     token_transition_count[y[i-1]] = {}
                # token_transition_count[y[i-1]][(x[i-1], x[i])] = token_transition_count[y[i-1]].get((x[i-1], x[i]), 0) + 1 # COMBINED FEATURE

                if bi_count.get((y[i-1], y[i])) == None:
                    bi_count[(y[i-1], y[i])] = {}
                bi_count[(y[i-1], y[i])][x[i]] = bi_count[(y[i-1], y[i])].get(x[i], 0) + 1
                bi_count[(y[i-1], y[i])][x[i-1]] = bi_count[(y[i-1], y[i])].get(x[i-1], 0) + 1    # BACKWARD BIGRAM

                state_count[(y[i-1], y[i])] = state_count.get((y[i-1], y[i]), 0) + 1

            # Middle words
            else:
                if emission_count.get(y[i]) == None:
                    emission_count[y[i]] = {}
                emission_count[y[i]][x[i]] = emission_count[y[i]].get(x[i], 0) + 1

                if uni_count.get(y[i]) == None:
                    uni_count[y[i]] = {}
                uni_count[y[i]][x[i-1]] = uni_count[y[i]].get(x[i-1], 0) + 1

                if uni_count.get(y[i]) == None:
                    uni_count[y[i]] = {}
                uni_count[y[i]][x[i+1]] = uni_count[y[i]].get(x[i+1], 0) + 1

                if transition_count.get(y[i-1]) == None:
                    transition_count[y[i-1]] = {}
                transition_count[y[i-1]][y[i]] = transition_count[y[i-1]].get(y[i], 0) + 1

                # if token_transition_count.get(y[i-1]) == None:
                #     token_transition_count[y[i-1]] = {}
                # token_transition_count[y[i-1]][(x[i-1], x[i])] = token_transition_count[y[i-1]].get((x[i-1], x[i]), 0) + 1  # COMBINED FEATURE

                if bi_count.get((y[i-1], y[i])) == None:
                    bi_count[(y[i-1], y[i])] = {}
                bi_count[(y[i-1], y[i])][x[i]] = bi_count[(y[i-1], y[i])].get(x[i], 0) + 1
                bi_count[(y[i-1], y[i])][x[i+1]] = bi_count[(y[i-1], y[i])].get(x[i+1], 0) + 1      # FORWARD BIGRAM
                bi_count[(y[i-1], y[i])][x[i-1]] = bi_count[(y[i-1], y[i])].get(x[i-1], 0) + 1      # BACKWARD BIGRAM

                state_count[(y[i-1], y[i])] = state_count.get((y[i-1], y[i]), 0) + 1

            state_count[y[i]] = state_count.get(y[i], 0) + 1

        state_count[(y[n-1], stop_state)] = state_count.get((y[n-1], stop_state), 0) + 1

    # return emission_count, transition_count, token_transition_count, uni_count, bi_count, state_count
    return emission_count, transition_count, uni_count, bi_count, state_count

def get_feature_dict_p6i(emission_count, transition_count, uni_count, bi_count, state_count):
    f = {}
    for tag, tokens in emission_count.items():
        for token, e_count in tokens.items():
            key = "emission: " + tag + '+' + token
            e_prob = np.log(e_count/state_count[tag])
            f[key] = e_prob

    for prev_tag, next_tags in transition_count.items():
        for next_tag, t_count in next_tags.items():
            key = "transition: " + prev_tag + '+' + next_tag
            t_prob = np.log(t_count/state_count[prev_tag])
            f[key] = t_prob
    
    # for prev_tag, next_tokens in token_transition_count.items():
    #     for next_tokens, tt_count in next_tokens.items():
    #         key = "token_transition: " + prev_tag + '+' + next_tokens[0] + '+' + next_tokens[1]
    #         tt_prob = np.log(tt_count/state_count[prev_tag])
    #         f[key] = tt_prob

    for tag, tokens in uni_count.items():
        for token, u_count in tokens.items():
            key = "unigram: " + tag + '+' + token
            u_prob = np.log(u_count/state_count[tag])
            f[key] = u_prob

    for tag, tokens in bi_count.items():
        for token, b_count in tokens.items():
            key = "bigram: " + tag[0] + '+' + tag[1] + '+' + token
            b_prob = np.log(b_count/state_count[tag])
            # print(b_prob)
            f[key] = b_prob

    return f

def viterbi_p6i(x, possible_states, f, default_index=0):
    n = len(x)
    d = len(possible_states)
    scores = np.full((n, d), -np.inf)
    bp = np.full((n, d), default_index, dtype=np.int32)

    for i in range(len(possible_states)):
        t_key = "transition: START"+possible_states[i]
        e_key = "emission: "+possible_states[i]+"+"+x[0]
        b_key1 = "bigram: START"+possible_states[i]+"+"+x[0]

        t_prob = f.get(t_key, -2**31)
        e_prob = f.get(e_key, -2**31)
        b_prob1 = f.get(b_key1, -2**31)
        
        if n > 1:
            u_key = "unigram: "+possible_states[i]+"+"+x[1]
            u_prob = f.get(u_key, -2**31)

            b_key2 = "bigram: START"+possible_states[i]+"+"+x[1]
            b_prob2 = f.get(b_key2, -2**31)

            # tt_key = "token_transition: " + possible_states[i] + '+' + x[0] + '+' + x[1]
            # tt_prob = f.get(tt_key, -2**31)

            # scores[0, i] = t_prob + e_prob + u_prob + b_prob1 + b_prob2 + tt_prob
            scores[0, i] = t_prob + e_prob + u_prob + b_prob1 + b_prob2

        else:
            scores[0, i] = t_prob + e_prob + b_prob1
    
    for i in range(1, n):
        for k in range(len(possible_states)):
            for j in range(len(possible_states)):
                t_key = "transition: "+possible_states[k]+"+"+possible_states[j]
                e_key = "emission: "+possible_states[j]+"+"+x[i]
                u_key1 = "unigram: "+possible_states[j]+"+"+x[i-1]
                b_key1 = "bigram: "+possible_states[k]+possible_states[j]+"+"+x[i]
                b_key2 = "bigram: "+possible_states[k]+"+"+possible_states[j]+"+"+x[i-1]

                t_prob = f.get(t_key, -2**31)
                e_prob = f.get(e_key, -2**31)
                u_prob1 = f.get(u_key1, -2**31)
                b_prob1 = f.get(b_key1, -2**31) 
                b_prob2 = f.get(b_key2, -2**31)

                if i != n-1:
                    u_key2 = "unigram: "+possible_states[j]+"+"+x[i+1]
                    u_prob2 = f.get(u_key2, -2**31)

                    b_key3 = "bigram: "+possible_states[k]+"+"+possible_states[j]+"+"+x[i+1]
                    b_prob3 = f.get(b_key3, -2**31)

                    # tt_key = "token_transition: " + possible_states[j] + '+' + x[i] + '+' + x[i+1]
                    # tt_prob = f.get(tt_key, -2**31)

                    # overall_score = e_prob + t_prob + tt_prob + u_prob1 + u_prob2 + b_prob1 + b_prob2 + b_prob3 + scores[i-1, k]
                    overall_score = e_prob + t_prob + u_prob1 + u_prob2 + b_prob1 + b_prob2 + b_prob3 + scores[i-1, k]

                else:
                    overall_score = e_prob + t_prob + u_prob1 + b_prob1 + b_prob2 + scores[i-1, k]

                if overall_score > scores[i, j]:
                    scores[i, j] = overall_score
                    bp[i,j] = k
    
    highest_score = -2**31
    highest_bp = default_index
    for i in range(len(possible_states)):
        t_key = "transition: "+possible_states[i]+"+STOP"
        t_prob = f.get(t_key, -2**31)

        b_key2 = "bigram: "+possible_states[i]+"+"+"+STOP"+"+"+x[n-1]
        b_prob2 = f.get(b_key2, -2**31)

        overall_score = t_prob + b_prob2 + scores[n-1, i]
        
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

def decode_p6i(path, states, f, output_filename):
    default_index = states.index('O')
    sentences = list()

    with open(path) as file:
        lines = file.readlines()
        sentence = list()
        for line in lines:
            formatted_line = line.strip()   
            
            if(len(formatted_line) ==0):
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            sentence.append(formatted_line)

    with open(output_filename, "w") as wf:
        for sentence in sentences:
            pred_sentence = viterbi_p6i(sentence, states, f, default_index)        
            for i in range(len(sentence)):
                wf.write(sentence[i] + " " + pred_sentence[i] + "\n")
                
            wf.write("\n")

if __name__ == '__main__':
    possible_states = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative', 'B-neutral', 'I-neutral']
    
    print('Reading train file...')
    train_dataset = read_train_file('./dataset/train')

    print('Getting features...')
    emission_count, transition_count, uni_count, bi_count, state_count = get_feature_count_p6i(train_dataset)
    f_p6i = get_feature_dict_p6i(emission_count, transition_count, uni_count, bi_count, state_count)

    print('Decoding dataset/dev.in...')
    decode_p6i("dataset/dev.in", possible_states, f_p6i, 'dataset/dev.p6i.out')

    print('Ran decoding on dataset/dev.in. Output: dataset/dev.p6i.out')

    print('Running evaluation using conlleval...')
    g_tags, p_tags = get_tags('dataset/dev.p6i.out', 'dataset/dev.out')
    print(evaluate(g_tags,p_tags,verbose=True))