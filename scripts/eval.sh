which python
answers_file="benchmark/outputs/main_table/ours/ours_7b.json"
model_type="ours" # ["ours", "qwen", "qwen2_5", "deepseek", "llava", "internvl", "internvl2", "internvl_X_2_5", "gpt4o", "gpto1"]
iou=0.65
cot_rec_err_threhold=50 # change based on the different model, we use 50 for models like ours (sve-math-deepseek+)/internvl2/internvl2_5 and 150 for models like qwen

python benchmark/evaluate_release.py \
        --gt_file benchmark/data_final_V2.0.0/annotation/FINAL_COMBINE_MIX_V2.0.0.json \
        --image_folder benchmark/data_final_V2.0.0 \
        --answers_file ${answers_file} \
        --model_type ${model_type} \
        --iou ${iou} \
        --cot_rec_err_threhold ${cot_rec_err_threhold}