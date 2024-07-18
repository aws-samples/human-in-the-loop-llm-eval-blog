import boto3
import json
import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter
from config import *
import os

# Bedrock initialization and call
bedrock_client = boto3.client(service_name='bedrock-runtime')

# get all csv files that dont end with a certain phrase in a folder in local
def get_csv_files(path, phrases, field):
    files = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            filename = file[:-4]
            for phrase in phrases:
                if filename.endswith(phrase):
                    if field == 'QnA':
                        if filename.startswith("nl_queries"): 
                            files.append(file)
                    else: 
                        if not filename.startswith("nl_queries"):
                            files.append(file)
                    break
    return files

## Initializing session state variables
#-------------------------------------------------------------------------------

# Human in the Loop Resulting Dataframe
if "result_comment_df_A" not in st.session_state:
    st.session_state.result_comment_df_A = \
        pd.DataFrame(columns=['question', 'output', 'comment'])

if "result_comment_df_B" not in st.session_state:
    st.session_state.result_comment_df_B = \
        pd.DataFrame(columns=['question', 'output', 'comment'])

# HITL Scores for Prompt A
if "result_df_A" not in st.session_state:
    st.session_state.result_df_A = \
        pd.DataFrame(columns=["fluency", "coherence", "creativity",
                              "toxicity", "relevance", "completeness",
                              "factuality", "overall_quality"])

# HITL Scores for Prompt B
if "result_df_B" not in st.session_state:
    st.session_state.result_df_B = \
        pd.DataFrame(columns=["fluency", "coherence", "creativity",
                              "toxicity", "relevance", "completeness",
                              "factuality", "overall_quality"])

# HITL Scores Variables Prompt A
if 'fluency_A' not in st.session_state:
    st.session_state.fluency_A = 'Not Applicable'

if 'coherence_A' not in st.session_state:
    st.session_state.coherence_A = 'Not Applicable'

if 'creativity_A' not in st.session_state:
    st.session_state.creativity_A = 'Not Applicable'

if 'toxicity_A' not in st.session_state:
    st.session_state.toxicity_A = 'Not Applicable'

if 'relevance_A' not in st.session_state:
    st.session_state.relevance_A = 'Not Applicable'

if 'completeness_A' not in st.session_state:
    st.session_state.completeness_A = 'Not Applicable'

if 'factuality_A' not in st.session_state:
    st.session_state.factuality_A = 'Not Applicable'

if 'overall_quality_A' not in st.session_state:
    st.session_state.overall_quality_A = 'Not Applicable'

# HITL Scores Variables Prompt B
if 'fluency_B' not in st.session_state:
    st.session_state.fluency_B = 'Not Applicable'

if 'coherence_B' not in st.session_state:
    st.session_state.coherence_B = 'Not Applicable'

if 'creativity_B' not in st.session_state:
    st.session_state.creativity_B = 'Not Applicable'

if 'toxicity_B' not in st.session_state:
    st.session_state.toxicity_B = 'Not Applicable'

if 'relevance_B' not in st.session_state:
    st.session_state.relevance_B = 'Not Applicable'

if 'completeness_B' not in st.session_state:
    st.session_state.completeness_B = 'Not Applicable'

if 'factuality_B' not in st.session_state:
    st.session_state.factuality_B = 'Not Applicable'

if 'overall_quality_B' not in st.session_state:
    st.session_state.overall_quality_B = 'Not Applicable'

if 'first_score' not in st.session_state:
    st.session_state.first_score = 0


# Accumulated score of whether the first output is better
if 'first_score' not in st.session_state:
    st.session_state.first_score = 0

# Accumulated score of whether the second output is better
if 'sec_score' not in st.session_state:
    st.session_state.sec_score = 0

# Row index from new_prompts.csv
if 'row_index' not in st.session_state:
    st.session_state.row_index = 0

# Feedback from the user
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# First Prompt Approval Rate
if 'first_prompt_score' not in st.session_state:
    st.session_state.first_prompt_score = 0

# Second Prompt Approval Rate
if 'sec_prompt_score' not in st.session_state:
    st.session_state.sec_prompt_score = 0

# Dictionary of options for the first output (ex: Correct, hallucination, etc..)
if 'first_label_map' not in st.session_state:
    first_counter = Counter(LABEL_LIST)
    st.session_state.first_label_map = \
        Counter({key: value - 1 for key, value in first_counter.items()})

# Dictionary of options for the second output (ex: Correct, hallucination, etc..)
if 'sec_label_map' not in st.session_state:
    sec_counter = Counter(LABEL_LIST)
    st.session_state.sec_label_map = \
        Counter({key: value - 1 for key, value in sec_counter.items()})
#-------------------------------------------------------------------------------


# create sidebar in streamlit application
with st.sidebar:
    field = st.selectbox(
        'Select Field to Analyze',
        list(metadata_prompts_qns.keys()) + ["QnA"])

    fileOptions = get_csv_files(INPUT_PATH, models_used, field)

    # listen to user prompt A selection
    first_file = st.selectbox(
        'Select First File for Comparison',
        fileOptions)

    prompt_name_first = first_file.split("_")[2]
    model_first = " ".join(first_file.split("_")[-2:])[:-4]

    # listen to user prompt B selection
    sec_file = st.selectbox(
        'Select Second File for Comparison',
        fileOptions)

    prompt_name_sec = sec_file.split("_")[2]
    model_second = " ".join(sec_file.split("_")[-2:])[:-4]

    # start button shown after model/param selected
    if first_file != 'None' and sec_file != "None":
        df_file_one = pd.read_csv(INPUT_PATH + first_file)
        df_file_two = pd.read_csv(INPUT_PATH + sec_file)

        # create a start button
        start_button = st.button("Start")

# update the windows
def update_windows(row_file_one, row_file_two):
    """
    Update text boxes in the streamlit app, primarily the questions,
        instructions and the outputs. Make sure file one and file two
        correspond to the same question-answer-context trio
    Args:
        row_file_one(Pandas table row): Particular question/answer pair to analyze for first file
        row_file_two(Pandas table row): Particular question/answer pair to analyze for second file
        first_prompt_id(str): either default, none, or a prompt index for the first output
        sec_prompt_id(str): either default, none, or a prompt index for the second output
    Returns:
        None
    """
    if field == 'QnA': 
        question_window.info(row_file_one['question'], icon="🤖")
    else:
        question_window.info(metadata_prompts_qns[field], icon="🤖")
    first_prompt_output = row_file_one['prompt']
    sec_prompt_output = row_file_two['prompt']

    first_prompt_window.info(first_prompt_output)
    sec_prompt_window.info(sec_prompt_output)
    context_window.info(row_file_one['context'])
    if field == 'QnA': # For the QnA Use Case
        first_response = row_file_one["Answers"]
        sec_response = row_file_two["Answers"]
        if "Ground_Truth" in row_file_one.keys(): 
            gt_window.info(row_file_one["Ground_Truth"])
        else: 
            gt_window.info("Ground Truth Not Available")
    else: # For the Summarization Use Case
        first_response = row_file_one[field]
        sec_response = row_file_two[field]
        if f'gt_{field}' in row_file_one.keys(): 
            gt_window.info(row_file_one[f'gt_{field}'])
        else: 
            gt_window.info("Ground Truth Not Available")
    first_output_window.info(first_response)
    sec_output_window.info(sec_response)


# update prompt approve scores and comparison scores
def update_scores():
    """
    Update all session state variables
    """
    if vote == 'Output A is better':
        st.session_state.first_score += 1
    elif vote == 'Output B is better':
        st.session_state.sec_score += 1

    if first_prompt_score == "Approve":
        st.session_state.first_prompt_score += 1

    if sec_prompt_score == "Approve":
        st.session_state.sec_prompt_score += 1


def calculate_score(score): 
    if not score or score == 'Not Applicable':
        return np.NaN
    else:
        return CHOICES_RESULTS[score]


def update_results_A():
    new_row = {
        "fluency": calculate_score(st.session_state.fluency_A),
        "coherence": calculate_score(st.session_state.coherence_A),
        "creativity": calculate_score(st.session_state.creativity_A),
        "toxicity": calculate_score(st.session_state.toxicity_A),
        "relevance": calculate_score(st.session_state.relevance_A),
        "completeness": calculate_score(st.session_state.completeness_A),
        "factuality":  calculate_score(st.session_state.factuality_A),
        "overall_quality": calculate_score(st.session_state.overall_quality_A)
    }
    st.session_state.result_df_A = pd.concat([st.session_state.result_df_A,
                                              pd.DataFrame([new_row])])


def update_results_B():
    new_row = {
        "fluency": calculate_score(st.session_state.fluency_B),
        "coherence": calculate_score(st.session_state.coherence_B),
        "creativity": calculate_score(st.session_state.creativity_B),
        "toxicity": calculate_score(st.session_state.toxicity_B),
        "relevance": calculate_score(st.session_state.relevance_B),
        "completeness": calculate_score(st.session_state.completeness_B),
        "factuality":  calculate_score(st.session_state.factuality_B),
        "overall_quality": calculate_score(st.session_state.overall_quality_B)
    }
    st.session_state.result_df_B = pd.concat([st.session_state.result_df_B,
                                              pd.DataFrame([new_row])])


# update result analysis table
def update_results(is_output_1):
    """
    Update the session state results dataframe
    Args:
        is_output_1(boolean): True if for first output else second output
    """
    first_output = 'placeholder 1'
    sec_output = 'placeholder 2'
    output = first_output if is_output_1 else sec_output
    if is_output_1:
        update_results_A()
    else:
        update_results_B()
    comment = first_comment if is_output_1 else sec_comment

    if comment:
        if field == 'QnA': 
            question = df_file_one.iloc[st.session_state.row_index]['question']
        else: 
            question = metadata_prompts_qns[field]
        new_row = {
            'question': question,
            'output': output,
            # 'labels': category,
            'comment': comment,
        }
        if is_output_1: 
            st.session_state.result_comment_df_A = pd.concat([st.session_state.result_comment_df_A,
                                                    pd.DataFrame([new_row])])
        else: 
            st.session_state.result_comment_df_B = pd.concat([st.session_state.result_comment_df_B,
                                                    pd.DataFrame([new_row])])


# set up title and captions of the webpage
st.title("Prompt Human Evaluation")
st.caption("This tool aims to provide an objective method for evaluating \
             the performance \
             of different prompts with a selected large language model. \
             The questions \
             for evaluation are chosen by considering the similarity of model \
             outputs' embeddings and toxicity scores.")

# define question and context window
question_window = st.empty()
question_window.info('Here is the input question ', icon="🤖")


# Prompt Windows
first_prompt_window = st.empty()
first_prompt_col, second_prompt_col = st.columns([0.5, 0.5])
with first_prompt_col: 
    st.subheader(f"Prompt for {model_first}")
    first_prompt_window = st.empty()
    first_prompt_window.info('Here is the prompt')
with second_prompt_col:
    st.subheader(f"Prompt for {model_second}")
    sec_prompt_window = st.empty()
    sec_prompt_window.info('Here is the prompt')
st.subheader("Context")
context_window = st.empty()
context_window.info('Here is the context. This can be a document, call transcript, etc..')

# define model output window
context_col = st.columns(1)
first_col, sec_col, gt_col = st.columns([0.333, 0.333, 0.333])
first_subheader = f"{model_first} output:"
sec_subheader = f"{model_second} output:"

with first_col:
    st.subheader(first_subheader)
    first_output_window = st.info('Here is the model output from using prompt template A')

with sec_col:
    st.subheader(sec_subheader)
    sec_output_window = st.info('Here is the model output from using prompt template B')

with gt_col:
    st.subheader("Ground Truth")
    gt_window = st.info('Here is the expected Answer')

# define evaluation window
first_form = st.form('First Form', clear_on_submit=True)

CHOICES = list(CHOICES_RESULTS.keys())

with first_form:
    vote = st.radio('Vote :thumbsup:',
            ('Output A is better', 'Tie', 'Output B is better'))
    first_input_col, sec_input_col = st.columns([0.5, 0.5])
    with first_input_col:
        first_prompt_score = st.radio('Do you approve output A?',
                                 ('Approve', 'Disapprove'))
        fluency_options_A = st.selectbox(
            '''**Fluency**: I find output A is structured, grammatically correct, 
            and linguistically coherent''', CHOICES, key='fluency_A'
        )        
        coherence_options_A = st.selectbox(
            '''**Coherence**: Output A is organized, well-structured, and easy
            to understand''',
            CHOICES, key='coherence_A'
        )        
        creativity_options_A = st.selectbox(
            '''**Creativity**: I find output A original, imaginative, and 
            unconventional''',
            CHOICES, key="creativity_A"
        )
        toxicity_options_A = st.selectbox(
            '''**Toxicity**: Output A may be perceived as rude, disrespectful, 
            and\or prejudiced''',
            CHOICES, key="toxicity_A"
        )
        relevance_options_A = st.selectbox(
            '''**Relevance**: I find output A's text that is pertinent to the given
            context, task, or topic''',
            CHOICES, key="relevance_A"
        )
        completeness_options_A = st.selectbox(
            '''**Completeness**: Output A provided the right amount of detail for
            the given question''',
            CHOICES, key="completeness_A"
        )
        factuality_options_A = st.selectbox(
            '''**Factuality**: Is output A factually correct?''',
            CHOICES, key="factuality_A"
        )
        overall_quality_options_A = st.selectbox(
            '''**Overall Quality**: Output A is high quality in general''',
            CHOICES, key="overall_quality_A"
        )
        first_comment = st.text_input(
            "Feedback A",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            placeholder='Your text input for the prompt A',
        )

    with sec_input_col:
        sec_prompt_score = st.radio('Do you approve output B?',
                                    ('Approve', 'Disapprove'))

        fluency_options_B = st.selectbox(
            '''**Fluency**: I find output B is structured, grammatically correct,
            and linguistically coherent''',
            CHOICES, key="fluency_B"
        )
        coherence_options_B = st.selectbox(
            '''**Coherence**: Output B is organized, well-structured, and easy
            to understand''',
            CHOICES, key="coherence_B"
        )
        creativity_options_B = st.selectbox(
            '''**Creativity**: I find output B original, imaginative, and 
            unconventional''',
            CHOICES, key="creativity_B"
        )
        toxicity_options_B = st.selectbox(
            '''**Toxicity**: Output B may be perceived as rude, disrespectful, 
            and\or prejudiced''',
            CHOICES, key="toxicity_B"
        )
        relevance_options_B = st.selectbox(
            '''**Relevance**: I find output B's text that is pertinent to the given
            context, task, or topic''',
            CHOICES, key="relevance_B"
        )
        completeness_options_B = st.selectbox(
            '''**Completeness**: Output B provided the right amount of detail for
            the given question''',
            CHOICES, key="completeness_B"
        )
        factuality_options_B = st.selectbox(
            '''**Factuality**: Is output B factually correct?''',
            CHOICES, key="factuality_B"
        )
        overall_quality_options_B = st.selectbox(
            '''**Overall Quality**: Output B is high quality in general''',
            CHOICES, key="overall_quality_B"
        )
        sec_comment = st.text_input(
            "Feedback B",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            placeholder='Your text input for the prompt B',
        )

    next_button = st.form_submit_button("Save results and Next :arrow_right:")

# create the end buttons
end_button = st.button("End and show scores")

# define start button clicking operation
if first_file != 'None' and sec_file != 'None' and start_button:
    st.session_state.row_index = 0
    st.session_state.first_score = 0
    st.session_state.sec_score = 0
    update_windows(df_file_one.iloc[st.session_state.row_index], 
                   df_file_two.iloc[st.session_state.row_index])


# Button click actions
if next_button:
    # print(st.session_state.fluency_A)
    # Get the next row in the dataframe
    update_results(True)
    update_results(False)
    update_scores()
    st.session_state.row_index += 1

    if st.session_state.row_index < len(df_file_one):
        update_windows(df_file_one.iloc[st.session_state.row_index], 
                   df_file_two.iloc[st.session_state.row_index])

# define end button clicking operation
if end_button:
    st.divider()
    st.subheader(":white_check_mark: Result Analysis")
    st.markdown("###### Prompt Comparison Result:")
    first_result_col, sec_result_col = st.columns([0.5, 0.5])
    total_score = st.session_state.row_index

    # display prompt comparison scores
    with first_result_col:
        st.write("Prompt A Score:", st.session_state.first_score)
        if total_score:
            st.write("Prompt A approve rate:",
                     str(round(st.session_state.first_prompt_score * 100 / total_score,
                               2)) + '%')
        st.write("Prompt A Fluency Mean:", st.session_state.result_df_A['fluency'].mean())
        st.write("Prompt A Fluency Median:", st.session_state.result_df_A['fluency'].median())
        st.write("Prompt A Coherence Mean:", st.session_state.result_df_A['coherence'].mean())
        st.write("Prompt A Coherence Median:", st.session_state.result_df_A['coherence'].median())
        st.write("Prompt A Creativity Mean:", st.session_state.result_df_A['creativity'].mean())
        st.write("Prompt A Creativity Median:", st.session_state.result_df_A['creativity'].median())
        st.write("Prompt A Toxicity Mean:", st.session_state.result_df_A['toxicity'].mean())
        st.write("Prompt A Toxicity Median:", st.session_state.result_df_A['toxicity'].median())
        st.write("Prompt A Relevance Mean:", st.session_state.result_df_A['relevance'].mean())
        st.write("Prompt A Relevance Median:", st.session_state.result_df_A['relevance'].median())
        st.write("Prompt A Completeness Mean:", st.session_state.result_df_A['completeness'].mean())
        st.write("Prompt A Completeness Median:", st.session_state.result_df_A['completeness'].median())
        st.write("Prompt A Factuality Mean:", st.session_state.result_df_A['factuality'].mean())
        st.write("Prompt A Factuality Median:", st.session_state.result_df_A['factuality'].median())
        st.write("Prompt A Overall Quality Mean:", st.session_state.result_df_A['overall_quality'].mean())
        st.write("Prompt A Overall Quality Median:", st.session_state.result_df_A['overall_quality'].median())


    with sec_result_col:
        st.write("Prompt B Score:", st.session_state.sec_score)
        if total_score:
            st.write("Prompt B approve rate:",
                     str(round(st.session_state.sec_prompt_score * 100 / total_score,
                               2)) + '%')
        st.write("Prompt B Fluency Mean:", st.session_state.result_df_B['fluency'].mean())
        st.write("Prompt B Fluency Median:", st.session_state.result_df_B['fluency'].median())
        st.write("Prompt B Coherence Mean:", st.session_state.result_df_B['coherence'].mean())
        st.write("Prompt B Coherence Median:", st.session_state.result_df_B['coherence'].median())
        st.write("Prompt B Creativity Mean:", st.session_state.result_df_B['creativity'].mean())
        st.write("Prompt B Creativity Median:", st.session_state.result_df_B['creativity'].median())
        st.write("Prompt B Toxicity Mean:", st.session_state.result_df_B['toxicity'].mean())
        st.write("Prompt B Toxicity Median:", st.session_state.result_df_B['toxicity'].median())
        st.write("Prompt B Relevance Mean:", st.session_state.result_df_B['relevance'].mean())
        st.write("Prompt B Relevance Median:", st.session_state.result_df_B['relevance'].median())
        st.write("Prompt B Completeness Mean:", st.session_state.result_df_B['completeness'].mean())
        st.write("Prompt B Completeness Median:", st.session_state.result_df_B['completeness'].median())
        st.write("Prompt B Factuality Mean:", st.session_state.result_df_B['factuality'].mean())
        st.write("Prompt B Factuality Median:", st.session_state.result_df_B['factuality'].median())
        st.write("Prompt B Overall Quality Mean:", st.session_state.result_df_B['overall_quality'].mean())
        st.write("Prompt B Overall Quality Median:", st.session_state.result_df_B['overall_quality'].median())

    # save the result dataframes
    OUTPUT_PATH = "comments_output.csv"
    if first_file[:-4] not in os.listdir(INPUT_PATH):
        os.mkdir(path=INPUT_PATH + first_file[:-4])
    if sec_file[:-4] not in os.listdir(INPUT_PATH):
        os.mkdir(path=INPUT_PATH + sec_file[:-4])
    OUTPUT_PATH_A = INPUT_PATH + first_file[:-4]+ "/" + first_file[:-4] + "_" + field + "_output.csv"
    OUTPUT_PATH_B = INPUT_PATH + sec_file[:-4] + "/" + sec_file[:-4] + "_"+ field + "_output.csv"
    st.session_state.result_comment_df_A.to_csv(INPUT_PATH + first_file[:-4]+ "/" + OUTPUT_PATH, index=False)
    st.session_state.result_comment_df_B.to_csv(INPUT_PATH + sec_file[:-4]+ "/" + OUTPUT_PATH, index=False)
    st.session_state.result_df_A.to_csv(OUTPUT_PATH_A, index=False)
    st.session_state.result_df_B.to_csv(OUTPUT_PATH_B, index=False)

# save human interaction
if len(st.session_state.result_comment_df_A):
    st.markdown(f"###### Saved Samples {model_first} Output Comments:")
    st.write(st.session_state.result_comment_df_A)
if len(st.session_state.result_comment_df_B):
    st.markdown(f"###### Saved Samples {model_second} Output Comments:")
    st.write(st.session_state.result_comment_df_B)