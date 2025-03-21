#model="InternVL2-8B"
#model="Llama-3.2-11B-Vision-Instruct"
#model="Ovis1.6-Gemma2-9B"
model="Qwen2-VL-7B-Instruct"
echo $model
for lang in en de es fr hi pt zh; do
  echo $lang;
  echo "  text only"
  python scripts/evaluate.py ~/data3/multi-modal-stance/logs/*/text_only/${model}/prompt_en/tweet_${lang}/version_0/seed_0/predictions/validation.csv
  python scripts/compute_significance.py --predictions1 ~/data3/multi-modal-stance/logs/*/text_only/${model}/prompt_en/tweet_${lang}/version_0/seed_0/predictions/validation.csv --predictions2 ~/data3/multi-modal-stance/logs/*/text_only/${model}/prompt_en/tweet_en/version_0/seed_0/predictions/validation.csv
  echo "  image only"
  python scripts/evaluate.py ~/data3/multi-modal-stance/logs/*/image_only/${model}/prompt_en/tweet_${lang}/version_0/seed_0/predictions/validation.csv
  python scripts/compute_significance.py --predictions1 ~/data3/multi-modal-stance/logs/*/image_only/${model}/prompt_en/tweet_${lang}/version_0/seed_0/predictions/validation.csv --predictions2 ~/data3/multi-modal-stance/logs/*/image_only/${model}/prompt_en/tweet_en/version_0/seed_0/predictions/validation.csv
  echo "  text + image"
  python scripts/evaluate.py ~/data3/multi-modal-stance/logs/*/${model}/prompt_en/tweet_${lang}/version_0/seed_0/predictions/validation.csv
  python scripts/compute_significance.py --predictions1 ~/data3/multi-modal-stance/logs/*/${model}/prompt_en/tweet_${lang}/version_0/seed_0/predictions/validation.csv --predictions2 ~/data3/multi-modal-stance/logs/*/${model}/prompt_en/tweet_en/version_0/seed_0/predictions/validation.csv
  echo
done

