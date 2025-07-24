CORPUS="books"

# FORGET="../data/$CORPUS/raw/forget.txt"
FORGET="../data/news/raw/forget.txt"

RETAIN="../data/$CORPUS/raw/retain1.txt"

TARGET_DIR="muse-bench/MUSE-Books_target"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"

MAX_LEN=2048
EPOCHS=1
LR='1e-5'
PER_DEVICE_BATCH_SIZE=8 # 4 GPUs
FT_EPOCHS=1
FT_LR='1e-5'

SEED=1


algo_list=('npo' 'npo_gdr')
# algo_list=('npo_gdr')
forget_portion_list=(0.05)
# forget_portion_list=(0.05 0.1 0.25 0.5 0.75)
# forget_portion_list=(0.05 0.1 0.25 0.5 0.75 1.0)

upsample_ratios=(2.0 5.0 10.0 15.0 20.0)


for algo in "${algo_list[@]}"; do
    for forget_portion in "${forget_portion_list[@]}"; do
        for upsample_ratio in "${upsample_ratios[@]}"; do
            out_dir="./ckpt/$CORPUS/${algo}_${forget_portion}_s${SEED}_u${upsample_ratio}"
            # if portion is 1.0 don't include seed in output directory
            if [ "$forget_portion" == "1.0" ]; then
                out_dir="./ckpt/$CORPUS/${algo}_${forget_portion}_u${upsample_ratio}"
            fi
            # if news in forget file name, include it in output directory
            if [[ "$FORGET" == *"news"* ]]; then
                out_dir="./ckpt/$CORPUS/${algo}_news_${forget_portion}_u${upsample_ratio}"
                echo "Output directory: $out_dir"
            fi
            # if upsample ratio is 1.0, don't include it in output directory
            CUDA_VISIBLE_DEVICES=0,1,2,3 python unlearn.py \
                --algo $algo \
                --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
                --data_file $FORGET --retain_data_file $RETAIN \
                --out_dir $out_dir \
                --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
                --forget_portion $forget_portion \
                --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
                --seed $SEED \
                --upsample $upsample_ratio
        done
    done
done
