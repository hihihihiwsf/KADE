import pickle
import random
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from util_causaltb import generate_data
from IPython import embed

model_dir = 'bert-base-uncased' # uncased better
tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)

def get_sem_eval(datasets, opt_file_name, mask=True):
    data_set = []

    for data in datasets:
        doc_name, words, span1, span2, rel = data

        sentence_s = words[:]
        sentence_t = words[:]

        event1 = np.take(sentence_s, span1, 0)
        event2 =  np.take(sentence_t, span2, 0)

        sentence_s_o = ' '.join(sentence_s)
        sentence_t_o = sentence_s_o

        event1 = ' '.join(event1)
        event2 = ' '.join(event2)

        sentence_s = ['[CLS]'] + sentence_s + ['[SEP]']
        sentence_t = ['[CLS]'] + sentence_t + ['[SEP]']

        span1 = list(map(lambda x: x+1, span1))
        span2 = list(map(lambda x: x+1, span2))

        sentence_vec_s = []
        sentence_vec_t = []

        span1_vec = []
        span2_vec = []
        for i, w in enumerate(sentence_s):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            if i in span1:
                span1_vec.extend(list(range(len(sentence_vec_s), len(sentence_vec_s) + len(xx))))

            sentence_vec_s.extend(xx)

        for i, w in enumerate(sentence_t):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            if i in span2:
                span2_vec.extend(list(range(len(sentence_vec_t), len(sentence_vec_t) + len(xx))))

            sentence_vec_t.extend(xx)

        if mask:
            for i in span1_vec:
                sentence_vec_s[i] = 103
            for i in span2_vec:
                sentence_vec_s[i] = 103

        data_set.append([doc_name, sentence_vec_s, sentence_vec_t, span1_vec, span2_vec, rel, sentence_s_o, sentence_t_o, event1, event2])

    with open(opt_file_name, 'wb') as f:
        pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    label_file = '/u/wusifan/CATENA/data/Causal-TempEval3-eval.txt'
    document_dir = '/u/wusifan/CATENA/data/TempEval3-eval_COL/'
    tempeval_results = generate_data(label_file, document_dir)
    
    # get_sem_eval(tempeval_results, 'causaltb/tempeval.pickle', mask=False)

    label_file = '/u/wusifan/CATENA/data/Causal-TimeBank.CLINK.txt'
    document_dir = '/u/wusifan/CATENA/data/Causal-TimeBank_COL/'
    causalTB_results = generate_data(label_file, document_dir)

    # get_sem_eval(causalTB_results, 'causaltb/causaltb.pickle', mask=False)
    get_sem_eval(causalTB_results, 'causaltb/causaltb_mask.pickle', mask=True)

    import random 
    random.shuffle(causalTB_results)
    l = int(len(causalTB_results) / 10 * 9)

    train_set = causalTB_results[:l]
    test_set = causalTB_results[l:]

    get_sem_eval(train_set, 'causaltb/causaltb_train.pickle', mask=False)
    get_sem_eval(test_set, 'causaltb/causaltb_test.pickle', mask=False)
    get_sem_eval(train_set, 'causaltb/causaltb_mask_train.pickle', mask=True)
    get_sem_eval(test_set, 'causaltb/causaltb_mask_test.pickle', mask=True)