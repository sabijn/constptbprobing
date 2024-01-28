from pathlib import Path

lca_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/constptbprobing/parsing-as-pretraining/exp_trees/distilbert-base-uncased/concat_lca/pred_labels_test.txt')
lev_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/constptbprobing/parsing-as-pretraining/exp_trees/distilbert-base-uncased/concat_lev/pred_labels_test.txt')
unary_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/constptbprobing/parsing-as-pretraining/exp_trees/distilbert-base-uncased/concat_unary/pred_labels_test.txt')
pos_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/constptbprobing/parsing-as-pretraining/exp_trees/test_pos_and_words.txt')

# with open(pos_path,'r') as f:
#     text_and_pos = f.read().splitlines()
#     wordsandpos = []
#     w_p_sent = [('-BOS-','-BOS-')]
#     for line in text_and_pos:
#         if len(line) == 0:
#             w_p_sent.append(('-EOS-','-EOS-'))
#             wordsandpos.append(w_p_sent)
#             w_p_sent = [('-BOS-','-BOS-')]
#             continue
#         [w,p] = line.split()
#         w_p_sent.append((w,p))
with open(pos_path,'r') as f:
    text_and_pos = f.read().splitlines()
    wordsandpos = []
    sent = []
    for line in text_and_pos:
        if line == '':
            wordsandpos.append(sent)
            sent = []
        else:
            [w,p] = line.split()
            sent.append((w,p))
with open(lca_path, 'r') as f:
    labels = f.read().splitlines()
    labels = [l.split() for l in labels]
with open(lev_path, 'r') as f:
    levels = f.read().splitlines()
    levels = [l.split() for l in levels]
with open(unary_path, 'r') as f:
    unaries = f.read().splitlines()
    unaries = [l.split() for l in unaries]

#assert all([len(l1) == (len(l2)+3) == (len(l3)+3) == len(l4)+3 for l1,l2,l3,l4 in zip(wordsandpos,labels,levels,unaries)])
for i, (l1, l2, l3, l4) in enumerate(zip(wordsandpos,labels,levels,unaries)):
    print(l1)
    print(l2)
    print(l3)
    print(l4)
    assert len(l1) == len(l2)+3 == len(l3)+3 == len(l4)+3, f'{i} {len(l1)} {len(l2) + 3} {len(l3) + 3} {len(l4) + 3}'
