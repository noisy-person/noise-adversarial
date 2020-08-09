#!/bin/bash

for e in {0.3,0.5,0.7,1.0} 

do

	for i in {1,2} 

	do
   		for j in {0.1,0.2,0.3,0.4} 
	       	do
		   	python main_AG_NEWS.py --noise_rate=$j --epsilon=$e
	   	done
	done
done
