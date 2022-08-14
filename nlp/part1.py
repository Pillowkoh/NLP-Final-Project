import numpy as np
from pprint import pprint

def read_train_file(filename):
    with open(filename, encoding='utf-8') as f:
        file_content = f.read()

    # Split the entire file into sentences. Output: List of sentences
    sentences = file_content.strip().split('\n\n')

    # Split each sentence into their token_tag pair
    # Output: List of sentences. Each sentence is a list of token_tag_pair
    token_tag_pairs = [i.split('\n') for i in sentences]

    # Separate each token_tag_pair into a list of [token, tag].
    # Output: [[[token, tag], [token, tag], ...], [[token, tag], [token, tag], ...], ...]
    for idx, sentence in enumerate(token_tag_pairs):
        token_tags = [i.rsplit(' ', maxsplit=1) for i in sentence]
        token_tag_pairs[idx] = token_tags

    return token_tag_pairs

def get_transition_emission_counts(train_dataset):
    emission_count = {}
    transition_count = {}
    state_count = {}
    possible_states = []

    transition_count['START'] = {}
    for sentence in train_dataset:
        prev_state = None
        
        for token, tag in sentence:
            if tag not in possible_states:
                possible_states.append(tag)

            if emission_count.get(token) == None:
                emission_count[token] = {}
            
            emission_count[token][tag] = emission_count[token].get(tag, 0) + 1

            if prev_state != None:
                if transition_count.get(prev_state) == None:
                    transition_count[prev_state] = {}
                transition_count[prev_state][tag] = transition_count[prev_state].get(tag, 0) + 1

            else:
                transition_count['START'][tag] = transition_count['START'].get(tag, 0) + 1
                state_count['START'] = state_count.get('START', 0) + 1

            state_count[tag] = state_count.get(tag, 0) + 1
            prev_state = tag

        transition_count[prev_state]['STOP'] = transition_count[prev_state].get('STOP', 0) + 1

    return transition_count, emission_count, state_count

# Part i: estimate emission probabilities and add them to f
def add_e_prob(f, emission_count, state_count):
    for token, tags in emission_count.items():
        for tag, e_count in tags.items():
            key = "emission: " + tag + '+' + token
            e_prob = np.log(e_count/state_count[tag])
            f[key] = e_prob

# Part ii: estimate transition probabilities and add them to f
def add_t_prob(f, transition_count, state_count):
    for prev_tag, next_tags in transition_count.items():
        for next_tag, t_count in next_tags.items():
            key = "transition: " + prev_tag + '+' + next_tag
            t_prob = np.log(t_count/state_count[prev_tag])
            f[key] = t_prob

if __name__ == '__main__':
    train_dataset = read_train_file('./dataset/train')
    transition_count, emission_count, state_count = get_transition_emission_counts(train_dataset)
    f = {}

    add_e_prob(f, emission_count, state_count)
    add_t_prob(f, transition_count, state_count)

    print('Feature dictionary: ')
    pprint(f)