from scipy.optimize import fmin_l_bfgs_b
from conlleval import evaluate, evaluate_conll_file
from part2 import *
from part3 import *

def compute_gradients_with_reg(train_dataset, f, possible_states, eta = 0):
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

    for k, v in f.items():
        feature_gradients[k] =feature_gradients.get(k,0) + 2*eta*f[k]

    return feature_gradients

def compute_crf_loss_with_reg(train_dataset, f, possible_states, eta=0):
    loss = 0

    for sentence in train_dataset:
        x = [token_tag_pair[0] for token_tag_pair in sentence]
        y = [token_tag_pair[1] for token_tag_pair in sentence]
    # for i in range(len(input_sequences)):
        first_term = calculate_score(x, y, f)
        _, alpha = forward_algorithm(x, f, possible_states)
        loss += (first_term - alpha) * -1

    reg_loss = 0
    for f_key in f:
        reg_loss += f[f_key]**2
    reg_loss = eta * reg_loss
    loss += reg_loss
    return loss

def callbackF(w):
    '''
    This function will be called by "fmin_l_bfgs_b"
    Arg:
        w: weights, numpy array
    '''
    loss = compute_crf_loss_with_reg(train_dataset,f,possible_states,0.1) 
    print('Loss: {0:.4f}'.format(loss))

def get_loss_grad(w, *args): 
    '''
    This function will be called by "fmin_l_bfgs_b"
    Arg:
        w: weights, numpy array
    Returns:
        loss: loss, float
        grads: gradients, numpy array
    '''
    train_dataset, f, possible_states = args
    for i,k in enumerate(f.keys()):
        f[k] = w[i]
    
    loss = compute_crf_loss_with_reg(train_dataset,f,possible_states,0.1)
    grads = compute_gradients_with_reg(train_dataset,f,possible_states,0.1)
    np_grads = np.zeros(len(f))
    for i,k in enumerate(f.keys()):
        np_grads[i] = grads[k]
    grads = np_grads
    return loss, grads

if __name__ == '__main__':
    possible_states = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative', 'B-neutral', 'I-neutral']
    
    print('Reading train file...')
    train_dataset = read_train_file('./dataset/train')

    print('Getting features...')
    f = {}
    transition_count, emission_count, state_count = get_transition_emission_counts(train_dataset)
    add_e_prob(f, emission_count, state_count)
    add_t_prob(f, transition_count, state_count)

    print('Training...')
    init_w = np.zeros(len(f))
    result = fmin_l_bfgs_b(get_loss_grad, init_w, args=(train_dataset,f,possible_states), pgtol=0.01, callback=callbackF)
    weight, loss, dictionary = result
    for idx, key in enumerate(f.keys()):
        f[key] = weight[idx]
    
    print('Decoding dataset/dev.in...')
    decode_file("dataset/dev.in", possible_states, f, 'partial/dev.p4.out')

    print('Running evaluation using conlleval...')
    g_tags, p_tags = eval('partial/dev.p4.out', 'dataset/dev.out')
    print(evaluate(g_tags,p_tags,verbose=True))
