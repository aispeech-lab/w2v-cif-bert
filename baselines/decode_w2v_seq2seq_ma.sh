. ../path.sh

gpu=$1
label_type=char
beam=5
DATA_DIR=data/ma/hkust_style_char
data_name=test
MODEL_PATH=exp/finetune_w2v_seq2seq_layer1_selfdecode5k_20k/checkpoint_best.pt
RESULT_DIR=exp/finetune_w2v_seq2seq_layer1_selfdecode5k_20k/decode_beam${beam}

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu \
python $MAIN_ROOT/examples/speech_recognition/infer.py $DATA_DIR \
    --gen-subset $data_name \
    --path $MODEL_PATH \
    --labels $label_type \
    --results-path $RESULT_DIR \
    --task audio_seq2seq --w2l-decoder seq2seq_decoder \
    --criterion cross_entropy_acc  --iscn \
    --beam ${beam} --remove-bpe $label_type --max-tokens 400000
