# VerySimpleChunker
## About
***VerySimpleChunker*** is an efficient, off-the-shelf shallow parser. 
Trained on the CONLL-2000 shared-task, it achieved 91.2% accuracy.
It has it’s strength in speed, thanks to high performance machine learning package Torch7 .
### Requirement
* lua ( > 5.2 )
* Torch7

### Downloading
`$git clone https://github.com/shimaokasonse/VerySimpleChunker`

`$cd VerySimpleChunker`

### Parsing
An example input (sample.txt) would look like as follows:

```text:sample.txt
This
is
a
sample
sentence
.

I
think
torch
7
is
one
of
the
best
deep
learning
frameworks
.


```

To parse this document, just run the following command:

`$th parse.lua sample.txt > output.txt`

The parsed document (output.txt) becomes:

```text:output.txt
This	B-NP	
is	B-VP	
a	B-NP	
sample	I-NP	
sentence	I-NP	
.	O	

	
I	B-NP	
think	B-VP	
torch	B-NP	
7	I-NP	
is	B-VP	
one	B-NP	
of	B-PP	
the	B-NP	
best	I-NP	
deep	I-NP	
learning	I-NP	
frameworks	I-NP	
.	O	

	

```

 That's it !!
