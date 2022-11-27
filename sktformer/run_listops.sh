function runexp {
#
gpu=${1}
task=${2}
model=${3}
chkpath=${4}
layers=${5}
lms=${6}         # lms (landmarks) is r in paper
k_conv=${7}
wsize=${8}       # wsize is w in paper
lr=${9}
wd=${10}
seed=${11}
steps=${12}
flags=${13}
echo model $model
echo steps $steps
flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
flags_print=${flags_print//--/_}

expname=${task}-${model}-l_${layers}-lr_${lr}-wd_${wd}-lms_${lms}-winsize_${wsize}-k_conv_${k_conv}-seed_${seed}${flags_print}-chkpath_${chkpath}
#


# parser.add_argument("--embedding_dim", default=64, type=int)
# parser.add_argument("--transformer_dim", default=64, type=int)
# parser.add_argument("--transformer_hidden_dim", default=128, type=int)
# parser.add_argument("--head_dim", default=32, type=int)
# parser.add_argument("--num_head", default=2, type=int)
# parser.add_argument("--pooling_mode", default="MEAN", type=str)
cmd=" python -m torch.distributed.launch --nproc_per_node=2 run_tasks.py --model ${model} --task ${task}
    --num_landmarks ${lms}   --conv_kernel_size ${k_conv}  --window_size ${wsize} ${flags}
     --learning_rate 1e-4  --weight_decay 0.01 --chk_path ${chkpath} --num_layers ${layers}  --max_seq_len 2048
    --dropout_prob 0.1 --attention_dropout 0.1 --rnn_dropout 0.1 --vocab_size 32
    --num_train_steps ${steps} --num_eval_steps 62 --eval_frequency 3000 --batch_size 16 --warmup 1000 
    --n_train_samples 96000 --n_dev_samples 2000 --n_test_samples 2000 --num_classes 10 
    --seed ${seed} 
    --epoch 10 
"
# --pooling_mode CLS --cls_token --embedding_dim 64 --transformer_dim 64 --head_dim 32 --num_head 2 --pooling_mode MEAN --cls_token --embedding_dim 512 --transformer_dim 512 --head_dim 64 --num_head 8


# config.emb_dim = 512
#   config.num_heads = 8
#   config.num_layers = 4
#   config.qkv_dim = 512
#   config.mlp_dim = 1024
# --cls_token --embedding_dim 128 --transformer_dim 128 --head_dim 32 --num_head 4
debug=1
if [ ${debug} -eq 0 ]; then
cmd="${cmd} --logging --expname ${expname}  > logs/${expname}.log 2>&1 "
else
cmd="${cmd} "
fi
#
echo logs/${expname}.log
#
eval ${cmd}

}
# runexp
# dynamicsoftmax_v7 SKTAttention dynamicsoftmax_v77
# The following hyperparameters correspond to Transformer-LS (w,r = 8,32) in the paper.
# One can change it to Transformer-LS (best) with lms=2, win_size=16
# runexp  gpu   task    model    layers  lms  k_conv  win_size lr   wd   seed   
# runexp   0    listops   dynamicsoftmax_v7    "LRA_chks/listops_fmmattention_1018"    2       32    -1      8     1e-4 1e-4  1234
#runexp   1    listops  htransformer      "LRA_chks/listops_htransformer_1018"    2       32    -1      8     1e-4 1e-4  1234
runexp   0    listops  attention_SKT      "LRA_chks/listops_dynamic_v77_1016"    2       4   -1      8     1e-4 1e-4   1234 30000
#runexp   5    listops  sigmoiddynamicsoftmax_v77      "LRA_chks/listops_sigmoid"    2       32    -1      8     1e-4  0.01 1234


