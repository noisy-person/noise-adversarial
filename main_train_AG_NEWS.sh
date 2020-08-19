#!/bin/bash

for e in {0.1,0.2} 

do

	for i in {1,2} 

	do
   		for j in {0.1,0.2,0.3,0.4,0.5,0.6,0.7} 
	       	do
		   	python main_AG_NEWS.py --noise_rate=$j --noise_mode=uni --epsilon=$e --GPU=3 --mode=transition
	   	done
	done
done
