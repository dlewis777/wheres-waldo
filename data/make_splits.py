import os
import sys
import subprocess
import random

seen = {}

rando = []
for i in range(286):
	rando.append(i)
	print(i)
random.shuffle(rando)
counter = 0
for filename in os.listdir('256/notwaldo'):
	
	seen[counter] = filename
	print(counter)
	counter += 1
for i in range(len(rando)):
	if i < 240:

		subprocess.Popen("cp 256/notwaldo/" + seen[rando[i]] + " augment/train/" + seen[rando[i]],shell=True)  		
	else:
		subprocess.Popen("cp 256/notwaldo/" + seen[rando[i]] + " augment/dev/" + seen[rando[i]],shell=True)

counter = 0 

rando2 = []
for i in range(32):
	rando2.append(i)

random.shuffle(rando2)

seen2 = {}
counter = 0
for filename in os.listdir('256/waldo'):
	
	seen2[counter] = filename
	counter += 1

for i in range(len(rando2)):

	if counter < 25:
	
 		subprocess.Popen("cp 256/waldo/" + seen[rando2[i]] + " augment/train/" + seen[rando2[i]], shell=True)  		
	else:
		subprocess.Popen("cp 256/waldo/" + seen[rando2[i]] + " augment/dev/" + seen[rando2[i]],shell=True)

