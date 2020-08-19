#!/bin/bash
for i in {1.5,2.0}

do
	for j in {0.85,0.9}
	do
		python main_DBpedia.py --noise_rate=$j --noise_mode=uni --epsilon=$i --GPU=0 --mode=adv
	done
done

python main_DBpedia.py --noise_rate=0.5 --noise_mode=uni --epsilon=0.05 --GPU=0 --mode=adv
