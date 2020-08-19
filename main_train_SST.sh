#!/bin/bash

for e in {0.65,0.85,1} 

do
   		for j in {0.45,0.47,0.5}
	       	do
		   	python main_SST.py --noise_rate=$j --noise_mode=uni --epsilon=$e --GPU=2 --mode=transition
	   	done
done
