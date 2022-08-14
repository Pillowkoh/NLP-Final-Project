# Repository Structure

```
.
├── dataset                   	# Training/Dev dataset folder
│   ├── dev.in					
│   ├── dev.out
│   └── train                 
├── test						         # Test dataset folder
│   ├── test.in
├── partial						      # Outputs folder
│   ├── dev.p2.out              
│   ├── dev.p4.out              
│   ├── dev.p5.out              
│   ├── test.p6.CRF.out             
│   └── test.p6.model.out
├── conlleval.py				      # Used for evaluation in part 1-6i
├── eval.py						      # Used for evaluation in part6ii
├── model.pt					      # Trained weights for part6ii
├── part1.py					
├── part2.py
├── part3.py
├── part4.py
├── part5.py
├── part6i.py
├── part6ii.py
├── README.md
└── Final_Report.pdf			      # Report for final project
```

# Instructions

### How to run?

1. Open the terminal and ensure that you are in the correct working directory.
   It should be something like `~./NLP-Final-Project`

2. To run the python codes for each part, use the following command to execute the respective python codes:
   `python <question_part_to_run>.py`

3. To evaluate all the output files for part 2,4 and 5, use the following command:
   `python eval.py`

4. To produce the test output files for part 6i and part 6ii, use the following command:
   `python eval.py <model_id> <input_file>`
   <model_id> :
   0 - model in part 6i 
   1 - model in part 6ii

   <input_file>:
    test/test.in