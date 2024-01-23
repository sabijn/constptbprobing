#!/bin/bash
model=$1
traincutoff=$2
testcutoff=$3
# Uncomment the following lines to run the experiments from scratch

# mkdir exp_chunk/
# echo "STEP: extract chunking labels training sets"
# echo "STEP: orig simple"
# python3 data_prep/extract_bies_from_ptb.py -ptb_notr ptb-train_orig.notrace -text_toks exp_chunk/train_bies_orig_text.txt -bies_labels exp_chunk/train_bies_orig_simple.txt -cutoff $traincutoff
# echo "STEP: orig detailed"
# python3 data_prep/extract_bies_from_ptb.py -ptb_notr ptb-train_orig.notrace -text_toks exp_chunk/train_bies_orig_text_with_phrase_labels.txt -bies_labels exp_chunk/train_bies_orig_with_phrase_labels.txt -cutoff $traincutoff -with_phrase_labels 

# echo "STEP: extract chunking labels test sets"
# echo "STEP: orig simple"
# python3 data_prep/extract_bies_from_ptb.py -ptb_notr ptb-test_orig.notrace -text_toks exp_chunk/test_bies_orig_text.txt -bies_labels exp_chunk/test_bies_orig_simple.txt -cutoff $testcutoff
# echo "STEP: orig detailed"
# python3 data_prep/extract_bies_from_ptb.py -ptb_notr ptb-test_orig.notrace -text_toks exp_chunk/test_bies_orig_text_with_phrase_labels.txt -bies_labels exp_chunk/test_bies_orig_with_phrase_labels.txt -cutoff $testcutoff -with_phrase_labels

# echo "STEP: replace some quotes in the text files"
# ./data_prep/replace-quotes-XLNet.sh exp_chunk/train_bies_orig_text.txt
# ./data_prep/replace-quotes-XLNet.sh exp_chunk/test_bies_orig_text.txt

# mkdir exp_chunk/$model/
# echo "STEP: extract representations from LM "$model
# echo "STEP: train, orig"
# python3 NeuroX/neurox/data/extraction/transformers_extractor.py --aggregation average $model exp_chunk/train_bies_orig_text.txt exp_chunk/$model/train_bies_orig_activations.hdf5

# echo "STEP: test, orig"
# python3 NeuroX/neurox/data/extraction/transformers_extractor.py --aggregation average $model exp_chunk/test_bies_orig_text.txt exp_chunk/$model/test_bies_orig_activations.hdf5

# echo "STEP: prep control task labels"
# echo "orig2orig"
# python3 data_prep/prepare_random_baseline_labels.py -text_train exp_chunk/train_bies_orig_text.txt -text_dev exp_chunk/test_bies_orig_text.txt -labels_train_in exp_chunk/train_bies_orig_simple.txt -labels_train_out exp_chunk/train_bies_orig2orig_simple_controltask.txt -labels_dev_out exp_chunk/test_bies_orig2orig_simple_controltask.txt
# python3 data_prep/prepare_random_baseline_labels.py -text_train exp_chunk/train_bies_orig_text.txt -text_dev exp_chunk/test_bies_orig_text.txt -labels_train_in exp_chunk/train_bies_orig_with_phrase_labels.txt -labels_train_out exp_chunk/train_bies_orig2orig_with_phrase_labels_controltask.txt -labels_dev_out exp_chunk/test_bies_orig2orig_with_phrase_labels_controltask.txt

echo "STEP: run experiments"

echo "STEP orig/orig simple"
python3 syntax_probing_experiments.py -out_dir exp_chunk/$model/orig2orig_simple/ -train_tokens exp_chunk/train_bies_orig_text.txt -train_labels exp_chunk/train_bies_orig_simple.txt -dev_tokens exp_chunk/test_bies_orig_text.txt -dev_labels exp_chunk/test_bies_orig_simple.txt -train_activations exp_chunk/$model/train_bies_orig_activations.hdf5 -dev_activations exp_chunk/$model/test_bies_orig_activations.hdf5 -no_detailed_analysis
sleep 2
echo "STEP with labels"
python3 syntax_probing_experiments.py -out_dir exp_chunk/$model/orig2orig_with_phrase_labels/ -train_tokens exp_chunk/train_bies_orig_text.txt -train_labels exp_chunk/train_bies_orig_with_phrase_labels.txt -dev_tokens exp_chunk/test_bies_orig_text.txt -dev_labels exp_chunk/test_bies_orig_with_phrase_labels.txt -train_activations exp_chunk/$model/train_bies_orig_activations.hdf5 -dev_activations exp_chunk/$model/test_bies_orig_activations.hdf5 -no_detailed_analysis

echo "STEP: CT experimentss"

echo "STEP CT orig/orig simple"
python3 syntax_probing_experiments.py -out_dir exp_chunk/$model/orig2orig_simple_ct/ -train_tokens exp_chunk/train_bies_orig_text.txt -train_labels exp_chunk/train_bies_orig2orig_simple_controltask.txt -dev_tokens exp_chunk/test_bies_orig_text.txt -dev_labels exp_chunk/test_bies_orig2orig_simple_controltask.txt -train_activations exp_chunk/$model/train_bies_orig_activations.hdf5 -dev_activations exp_chunk/$model/test_bies_orig_activations.hdf5 -no_detailed_analysis
sleep 2
echo "STEP: with labels"
python3 syntax_probing_experiments.py -out_dir exp_chunk/$model/orig2orig_with_phrase_labels_ct/ -train_tokens exp_chunk/train_bies_orig_text.txt -train_labels exp_chunk/train_bies_orig2orig_with_phrase_labels_controltask.txt -dev_tokens exp_chunk/test_bies_orig_text.txt -dev_labels exp_chunk/test_bies_orig2orig_with_phrase_labels_controltask.txt -train_activations exp_chunk/$model/train_bies_orig_activations.hdf5 -dev_activations exp_chunk/$model/test_bies_orig_activations.hdf5 -no_detailed_analysis






