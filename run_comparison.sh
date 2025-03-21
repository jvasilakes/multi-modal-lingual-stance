langs=(en de es fr hi pt zh)
nlangs=${#langs[@]}
base_indir="/home/ac1jv/data3/multi-modal-stance/logs/"
base_outdir="/home/ac1jv/data3/multi-modal-stance/comparison_results/prompt_en/"
#indir=  # If empty indir will be /home/ac1jv/data3/multi-modal-stance/logs/${ds}/${model}/
indir=""  # ${base_indir}/${dataset}/${indir}/${model}
outdir="/home/ac1jv/data3/multi-modal-stance/comparison_results/${indir}"
mkdir -p $outdir
for model in "InternVL2-8B" "Llama-3.2-11B-Vision-Instruct" "Qwen2-VL-7B-Instruct" "Ovis1.6-Gemma2-9B"; do
  echo $model
  for ds in mccq mruc mtse mtwq mwtwt; do
    echo -n "  $ds "
    comp_outdir="${outdir}/${model}/${ds}/"
    mkdir -p "$comp_outdir"
    comp_indir="${base_indir}/${ds}/${indir}/${model}/"
    let end_i=nlangs-1
    for i in $(seq 0 ${end_i}); do
      let start=i+1
      for j in $(seq ${start} ${end_i}); do
	lang1=${langs[i]}
	lang2=${langs[j]}
        if [[ "${lang1}" == "${lang2}" ]]; then
          continue
        fi
	#if [[ "${lang1}" == "en" ]]; then
	#  first_arg="${comp_indir}/${lang1}/version_0/seed_0/predictions/validation.csv"
        #else
	first_arg="${comp_indir}/prompt_en/tweet_${lang1}/version_0/seed_0/predictions/validation.csv"
	#fi
	echo -n '.'
	outfile="${comp_outdir}/${lang1}_${lang2}.json"
	python scripts/compare_predictions.py ${first_arg} ${comp_indir}/prompt_en/tweet_${lang2}/version_0/seed_0/predictions/validation.csv > $outfile
      done
    done
    echo
  done
done
