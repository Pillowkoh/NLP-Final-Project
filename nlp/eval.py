import sys
import torch
from conlleval import evaluate, evaluate_conll_file
from part1 import *
from part2 import get_tags
from part6i import *
from part6ii import *

if __name__ == '__main__':
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) < 2:
        print('Printing evaluations for Parts 2, 4, 5, 6i and 6ii. Ensure that output files are in dataset directory')

        for part in ['2', '4', '5', '6i', '6ii']:
            print('Part {}'.format(part))
            pred_fn = 'dataset/dev.p' + part + '.out'

            g_tags, p_tags = get_tags(pred_fn, 'dataset/dev.out')
            print(evaluate(g_tags,p_tags,verbose=True))
            print('\n')
    else:
        model_id, test_file = sys.argv[1], sys.argv[2]
        print('Decoding {}'.format(test_file))
        if model_id == 0:
            train_dataset = read_train_file('./dataset/train')
            emission_count, transition_count, token_transition_count, uni_count, bi_count, state_count = get_feature_count_p6i(train_dataset)
            f_p6i = get_feature_dict_p6i(emission_count, transition_count, token_transition_count, uni_count, bi_count, state_count)

            decode_p6i(test_file, possible_states, f_p6i, 'test/test.p6.CRF.out')
        elif model_id == 1:
            train_dataset = read_train_file('./dataset/train')
            train_tokens = [[token for token, tag in sent] for sent in train_dataset]
            train_tags= [[tag for token, tag in sent] for sent in train_dataset]
            train_vocab, train_tags_vocab = build_vocab(train_tokens), build_vocab(train_tags)
            train_vocab.set_default_index(train_vocab['a'])
            special_tags = (train_vocab['START'], train_vocab['STOP'], train_vocab['PAD'])
            
            model = torch.load('model.pt')
            predict_dev_in(model, test_file, 'test/test.p6.model.out', train_vocab, train_tags_vocab)





