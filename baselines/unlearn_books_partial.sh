CORPUS="books"

FORGET="../data/$CORPUS/raw/books_forget.csv"
RETAIN="../data/$CORPUS/raw/retain1.txt"

TARGET_DIR="muse-bench/MUSE-Books_target"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"

MAX_LEN=2048
EPOCHS=5
LR='1e-5'
PER_DEVICE_BATCH_SIZE=8 # 4 GPUs
FT_EPOCHS=5
FT_LR='1e-5'

SEED=1


algo_list=('npo' 'npo_gdr' 'npo_klr')
forget_portion_list=(0.05 0.1 0.25 0.5 0.75)


for algo in "${algo_list[@]}"; do
    for forget_portion in "${forget_portion_list[@]}"; do
        python unlearn.py \
            --algo $algo \
            --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
            --data_file $FORGET --retain_data_file $RETAIN \
            --out_dir "./ckpt/$CORPUS/${algo}_${forget_portion}_s${SEED}" \
            --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
            --forget_portion $forget_portion \
            --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
            --seed $SEED
    done
done
