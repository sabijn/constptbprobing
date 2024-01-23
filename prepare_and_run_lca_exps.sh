#!/bin/bash

model=$1
trainsamplesize=$2
cutoff=$3
mkdir exp_lca/

# echo "STEP: extract LCA labels"
# echo "STEP: orig"
# python3 data_prep/extract_lca_from_ptb.py -ptb_notr data/PennTreebank/ptb-train_orig.notrace -text_toks exp_lca/train_lca_orig_text.txt -rel_toks exp_lca/train_lca_orig_rel_toks.txt -rel_labels exp_lca/train_lca_orig_rel_labels.txt
# python3 data_prep/extract_lca_from_ptb.py -ptb_notr data/PennTreebank/ptb-test_orig.notrace -text_toks exp_lca/test_lca_orig_text.txt -rel_toks exp_lca/test_lca_orig_rel_toks.txt -rel_labels exp_lca/test_lca_orig_rel_labels.txt -cutoff $cutoff -max_sent_length 20

# echo "STEP: replace quotes"
# ./data_prep/replace-quotes-XLNet.sh exp_lca/train_lca_orig_text.txt
# ./data_prep/replace-quotes-XLNet.sh exp_lca/test_lca_orig_text.txt

# echo "STEP: sample training data"
# python3 data_prep/sample_training_data.py -rel_toks_i exp_lca/train_lca_orig_rel_toks.txt -labels_i exp_lca/train_lca_orig_rel_labels.txt -rel_toks_sampled exp_lca/train_lca_orig_rel_toks_sampled.txt -labels_sampled exp_lca/train_lca_orig_rel_labels_sampled.txt -target_size $trainsamplesize

echo "STEP: create control task labels"
echo "STEP: orig/orig"
python3 data_prep/prepare_random_baseline_labels.py -text_train exp_lca/train_lca_orig_text.txt -text_dev exp_lca/test_lca_orig_text.txt -labels_train_in exp_lca/train_lca_orig_rel_labels.txt -labels_train_out exp_lca/train_lca_orig2orig_rel_labels_ct.txt -labels_dev_out exp_lca/test_lca_orig2orig_rel_labels_ct.txt -mode pair

echo "STEP: sample ct train data"

echo "STEP: orig/orig"
python3 data_prep/sample_training_data.py -rel_toks_i exp_lca/train_lca_orig_rel_toks.txt -labels_i exp_lca/train_lca_orig2orig_rel_labels_ct.txt -rel_toks_sampled exp_lca/train_lca_orig2orig_rel_toks_ct_sampled.txt -labels_sampled exp_lca/train_lca_orig2orig_rel_labels_ct_sampled.txt -target_size $trainsamplesize

mkdir exp_lca/$model/
echo "STEP: extract representations from LM "$model
echo "STEP: train, orig"
python3 NeuroX/neurox/data/extraction/transformers_extractor.py --aggregation average $model exp_lca/train_lca_orig_text.txt exp_lca/$model/train_orig_activations.hdf5

echo "STEP: test, orig"
python3 NeuroX/neurox/data/extraction/transformers_extractor.py --aggregation average $model exp_lca/test_lca_orig_text.txt exp_lca/$model/test_orig_activations.hdf5

echo "STEP: combine activations"
for combtype in "concat"
do
    echo "STEP: orig "$combtype
    python3 data_prep/combine_activations.py -i exp_lca/$model/train_orig_activations.hdf5 -o exp_lca/$model/train_orig_$combtype\_activations.hdf5 -rel_toks exp_lca/train_lca_orig_rel_toks_sampled.txt -m $combtype -sampled
    python3 data_prep/combine_activations.py -i exp_lca/$model/test_orig_activations.hdf5 -o exp_lca/$model/test_orig_$combtype\_activations.hdf5 -rel_toks exp_lca/test_lca_orig_rel_toks.txt -m $combtype -sampled

    echo "STEP: CT orig2orig "$combtype
    python3 data_prep/combine_activations.py -i exp_lca/$model/train_orig_activations.hdf5 -o exp_lca/$model/train_orig2orig_$combtype\_ct_activations.hdf5 -rel_toks exp_lca/train_lca_orig2orig_rel_toks_ct_sampled.txt -m $combtype -sampled
done

echo "STEP: experiments"
for combtype in "concat"
do
    echo "STEP: orig/orig "$combtype
    python3 syntax_probing_experiments.py -out_dir exp_lca/$model/orig2orig_$combtype/ -train_tokens exp_lca/train_lca_orig_rel_toks_sampled.txt -train_labels exp_lca/train_lca_orig_rel_labels_sampled.txt -dev_tokens exp_lca/test_lca_orig_rel_toks.txt -dev_labels exp_lca/test_lca_orig_rel_labels.txt -train_activations exp_lca/$model/train_orig_$combtype\_activations.hdf5 -dev_activations exp_lca/$model/test_orig_$combtype\_activations.hdf5 -no_detailed_analysis 

    echo "STEP: control task experiments "
    echo "STEP: orig/orig "$combtype
    python3 syntax_probing_experiments.py -out_dir exp_lca/$model/orig2orig_$combtype\_ct/ -train_tokens exp_lca/train_lca_orig2orig_rel_toks_ct_sampled.txt -train_labels exp_lca/train_lca_orig2orig_rel_labels_ct_sampled.txt -dev_tokens exp_lca/test_lca_orig_rel_toks.txt -dev_labels exp_lca/test_lca_orig2orig_rel_labels_ct.txt -train_activations exp_lca/$model/train_orig2orig_$combtype\_ct_activations.hdf5 -dev_activations exp_lca/$model/test_orig_$combtype\_activations.hdf5 -no_detailed_analysis 
done
