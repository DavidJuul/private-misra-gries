# Differentially private Misra-Gries in practice

This project implements algorithms for creating Misra-Gries sketching, merging sketches, and for approximately and purely privatizing the sketches under differential privacy. This was done for a Bachelor's Project in Computer Science at the University of Copenhagen.

## Usage

The Python library requirements are listed in `requirements.txt` and can be installed with this command:
```
pip install -r requirements.txt
```

A non-private Misra-Gries sketch and an approximately (epsilon, delta)- or purely (epsilon, 0)-differentially private sketch can be created with these commands, respectively:
```
./pmg.py <sketch size> <epsilon> <delta> <stream file> [output sketch file]
./pmg.py <sketch size> <epsilon> 0 <universe size> <stream file> [output sketch file]
```
The input stream file must consist of non-negative integer elements each placed on their own line. The output non-private sketch of the given size can optionally be saved to a given output file in the JSON format. For pure privacy, the universe size, i.e. the integer one higher than the maximum possible integer element, must also be given.

A non-private merged sketch and an approximately (epsilon, delta)- or purely (epsilon, 0)-differentially private version of the merged sketch can be created with these commands, respectively:
```
./pmg.py merge <sketch size> <epsilon> <delta> <sketch file> [<sketch file> ...]
./pmg.py merge <sketch size> <epsilon> 0 <universe size> <sketch file> [<sketch file> ...]
```
The sketch files must be in the JSON format like those output by the sketch creation commands above.

A non-private sketch and an approximately (epsilon, delta)- or purely (epsilon, 0)-differentially private sketch in the user-level setting can be created with these commands, respectively:
```
./pmg.py userlevel <sketch size> <epsilon> <delta> <user element count> <stream file>
./pmg.py userlevel <sketch size> <epsilon> 0 <user element count> <universe size> <stream file>
```
The input stream files must be like before, except each user is allowed to have contributed up to a number of lines corresponding to the given user element count.

## Evaluation

A series of practical evaluations with unit tests, runtime benchmarks, and stochastic privacy tests were implemented. These can all be executed with this command:
```
./evaluate.py
```
