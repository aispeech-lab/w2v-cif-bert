gpu=$1
SAVE_DIR=exp/finetune_w2v_nar_en_layer4
W2V_PATH=../libri/wav2vec2_small.pt
DATA_DIR=data/en/subword
label_type=subword

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--train-subset train --valid-subset dev --no-epoch-checkpoints \
--labels $label_type --num-workers 2 --max-update 80000 \
--arch w2v_nar --task audio_ctc --criterion nar_qua_ctc_ce --best-checkpoint-metric uer \
--w2v-path $W2V_PATH --lambda-ctc 0.0 --lambda-qua 0.5 \
--decoder-embed-dim 768 --decoder-ffn-embed-dim 3072 --decoder-layers 4 --decoder-layerdrop 0.0 \
--decoder-attention-heads 8 --decoder-learned-pos --decoder-normalize-before \
--decoder-dropout 0.1 --decoder-attention-dropout 0.1 --decoder-activation-dropout 0.1 \
--apply-mask --mask-selection static --mask-other 0 --mask-length 2 --mask-prob 0.1 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.1 \
--feature-grad-mult 0.0 --freeze-finetune-updates 500 \
--validate-after-updates 30000  --validate-interval 4 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 1e-04 --lr-scheduler tri_stage \
--warmup-steps 10000 --hold-steps 20000 --decay-steps 30000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 \
--attention-dropout 0.0 --max-tokens 1000000 --seed 2337 --ddp-backend no_c10d --update-freq 1 \
--log-interval 300 --log-format simple --save-interval 1
