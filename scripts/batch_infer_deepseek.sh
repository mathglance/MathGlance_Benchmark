model_path="./checkpoints/deepseek-small"
answers_file="./benchmark/outputs/deepseek2vl-small.json"

CHUNKS=1
for IDX in 0; do
    CUDA_VISIBLE_DEVICES=$IDX python benchmark/infer_deepseek_final.py \
        --model-path ${model_path} \
        --prompt none \
        --question-file ./benchmark/data/annotation/FINAL_COMBINE_MIX_V2.0.0.json \
        --image-folder ./benchmark/data \
        --answers-file ${answers_file}\
        --num-chunks $CHUNKS \
        --chunk_size 512 \
        --temperature 0 \
        --top_p None \
        --repetition_penalty 1.1 \
        --chunk-idx $IDX &
done

wait

python benchmark/merge_pred.py --answers_file ${answers_file} --num_chunks $CHUNKS