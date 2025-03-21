#for model in internvl2 llama ovis qwen; do
for model in ovis; do
  #for dataset in mccq mruc mtse mtwq mwtwt; do
  for dataset in mtwq; do
    for lang in en; do
      python run.py predict --split validation configs/${dataset}/${model}/${lang}/covered_image_nontext/0shot.yaml
      python run.py predict --split validation configs/${dataset}/${model}/${lang}/covered_image_nontext/0shot_imageonly.yaml
    done
  done
done
