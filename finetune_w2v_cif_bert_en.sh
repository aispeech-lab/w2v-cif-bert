gpu=$1
W2V_PATH=../libri/wav2vec2_small.pt
BERT='bert-base-uncased'
SAVE_DIR=exp/finetune_w2v_cif_bert_en
DATA_DIR=data/en/bert_style
label_type=word

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--train-subset train --valid-subset dev --no-epoch-checkpoints \
--labels $label_type --num-workers 4 --max-update 80000 \
--lambda-ctc 2.0 --lambda-qua 0.2 --lambda-am 0.8 --lambda-lm 0.2 \
--arch w2v_cif_bert --task audio_cif_bert --criterion nar_qua_ctc_ce --best-checkpoint-metric uer \
--w2v-path $W2V_PATH --bert-name $BERT --infer-threash 0.8 \
--apply-mask --mask-selection static --mask-other 0 --mask-length 2 --mask-prob 0.1 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 \
--feature-grad-mult 0.0 --freeze-finetune-updates 500 --freeze-lm-finetune-updates 1000 \
--gold-rate-steps '(1000, 8000)' --gold-rate-range '(0.9, 0.2)' \
--validate-after-updates 7000  --validate-interval 4 --save-interval 4 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 4e-05 --lr-scheduler tri_stage \
--warmup-steps 5000 --hold-steps 10000 --decay-steps 30000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 \
--attention-dropout 0.0 --max-tokens 1000000 --seed 2337 --ddp-backend no_c10d --update-freq 1 \
--log-interval 200 --log-format simple
