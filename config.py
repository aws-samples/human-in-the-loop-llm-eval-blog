# constants
LABEL_LIST = ['Good', 'Hallucination', 'Toxic Content', 'Low Fluency',
              'Low Factual Consistency']
CHOICES_RESULTS = {'Not Applicable': 0, 'Strongly Disagree': 0, 'Disagree': 0.25,
           'Neither Agree nor Disagree': 0.5,
           'Agree': 0.75, 'Strongly Agree': 1}

INPUT_PATH = 'data/'
models_used = ["claude_instant", "haiku"]
# add more models inside models_used above. If you do, make sure the data file name includes
#    the name of the model at the end. For example, if the file is called summarization.csv, 
#    make sure to append the model name to be summarization_claude_instant.csv 
#    or summarization_haiku.csv

# summarization (non-QnA) questions
metadata_prompts_qns = {
    "summary": "What is the summary of the transcript?"
}
