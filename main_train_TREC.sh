#!/bin/bash

for e in {0.1,0.2} 

do


   		for j in {0.1,0.2,0.3,0.4,0.5,0.6,0.7} 
	       	do
		   	python main_TREC.py --noise_rate=$j --noise_mode=rand --epsilon=$e --GPU=2 --mode=transition
	   	done
done
