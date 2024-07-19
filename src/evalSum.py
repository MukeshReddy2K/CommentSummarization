import os
import numpy as np
import pandas as pd
import ollama
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer

#------------------------------------------------------------------------
def load_data_files(_DATA_FILE_QUESTIONS_, _DATA_FILE_RESPONSES_):
    # Load questions
    df_questions = pd.read_csv(_DATA_FILE_QUESTIONS_)
    # Load responses
    df_responses = pd.read_csv(_DATA_FILE_RESPONSES_)

    return df_questions, df_responses

#------------------------------------------------------------------------
def print_eval_questions_stats(df_questions):
    # Inspect questions dataframe
    print("\nQuestions dataframe:")
    print(df_questions.info())
    print("-"*50)
    # number of rows
    print("Number of questions: ", len(df_questions))
    print("-"*50)
    print(df_questions.head())
    print("-"*50)


    # questions stats
    print("\nQuestions stats:")
    # how many different forms are there (unique FormNumber)
    print("Number of unique forms: ", df_questions['FormNumber'].nunique())
    # min # of questions for a form
    print("Min # of questions for a form: ", df_questions.groupby('FormNumber').size().min())
    # max # of questions for a form
    print("Max # of questions for a form: ", df_questions.groupby('FormNumber').size().max())
    # avg # of questions for a form
    print("Avg # of questions for a form: ", df_questions.groupby('FormNumber').size().mean())

#------------------------------------------------------------------------
def print_eval_responses_stats(df_responses):
    # list distinct acad groups
    print("\nDistinct academic groups:")
    print(df_responses['AcadGroup'].unique())

    # list distinct subjects
    print("\nDistinct subjects:")
    print(df_responses['Subject'].unique())


    # now we need to do a count 
    # also for each acad unit, how many number of subjects
    # and for each subject, how many number of catalogs
    # and then for each catalog, how many number of sections

    # Acad units count: Colleges
    cnt_acad_units = len(df_responses['AcadGroup'].unique())
    print(f"Distinct academic groups: {cnt_acad_units}")

    # Subjects count : Departments
    # list distinct subjects
    cnt_subjects = len(df_responses['Subject'].unique())
    print(f"Distinct subjects: {cnt_subjects}")

    # Catalogs count : Courses
    # list distinct subject+catalogs
    cnt_catalogs = df_responses.groupby(['Subject', 'Catalog']).size()
    print(f"Distinct subject+catalogs: {len(cnt_catalogs)}")

    # Sections count
    # list distinct subject+catalog+ClassNbr
    cnt_sections = df_responses.groupby(['Subject', 'Catalog', 'ClassNbr']).size()
    print(f"Distinct subject+catalog+sections: {len(cnt_sections)}")

    # distinct form numbers
    cnt_forms = len(df_responses['EvaluationForm'].unique())    
    print(f"Distinct forms: {cnt_forms}")

#------------------------------------------------------------------------
def print_df_eval_response_forms(df_questions, df_responses):
    # print all the distinct forms in the df_responses
    print("\nDistinct forms:")
    forms_in_responses = df_responses['EvaluationForm'].unique()
    forms_in_responses.sort()
    for form in forms_in_responses:
        print(form)
        # check to see if it exists in df_questions
        # if so print the questions, otherwise print "No form found"
        if form in df_questions['FormNumber'].unique():
            # print(df_questions.loc[df_questions['FormNumber'] == form]['Question'])
            print('Form found')
        else:
            print("No form found")
#------------------------------------------------------------------------
def get_summary(model, comments):
    # interact with ollama api

    prompt = "Summarize the comments"
    prompt_comments = prompt + comments
    response = ollama.chat(model=model, messages=[
                            {
                                'role': 'user',
                                'content': prompt_comments,
                            },
                            ])
    
    return response['message']['content']
#------------------------------------------------------------------------
# now we need to iterate through each section
# get the form number for each section
# look up the questions for each section df_questions
# then we can get the responses for each question for each section df_responses
# for each response we can do sentiment analysis - use hugging face transformers library
# then we count how many are +ve, -ve and neutral
# then we concatenate all the comments for the question for each section
# and then summarize the comments for each question for each section
# then we write it to a file
def process_section(instructor, subject, catalog, section_nunber, df_section_responses, df_questions, _DATA_OUT_):
    print("="*88)
    print(f"Section: {section_nunber}")
    # print(df_section_responses.head())
    print("-"*50)
    print(f"Number of responses: {len(df_section_responses)}")
    # does it have a form number?
    form_numbers = df_section_responses['EvaluationForm'].unique()
    if len(form_numbers) == 0:
        print("No form number found")
        return
    elif len(form_numbers) > 1:
        print("Multiple form numbers found")
        return
    else:
        form_number = form_numbers[0]
        print(f"Form number: {form_number}")
        # get the questions for the form number
        df_form_questions = df_questions.loc[df_questions['FormNumber'] == form_number]
        print(f"Number of questions: {len(df_form_questions)}")
        print(f"Questions: {df_form_questions}")

        # for each question, get the responses
        # comments column is the one we are interested in
        # but naming of column is not clear, so we will use the index
        comment_starting_index = 12
        comment_iter = 0
        for index, row in df_form_questions.iterrows():
            print(f"index = {index}")
            question = row['Question']
            print(f"Question: {question}")
            # get the responses for the question
            # get the column name
            column_name = df_section_responses.columns[comment_starting_index + comment_iter]

            # bump comment_iter so for next question, we pick the next column
            comment_iter += 1

            print(f"Column name: {column_name}")
            # get the responses
            responses = df_section_responses[column_name]
            print(f"Number of responses: {len(responses)}")

            # clean out the responses
            # remove nan, empty strings
            responses = responses.dropna()
            responses = responses[responses != '']

            print(f"Number of responses after cleaning: {len(responses)}")
            # print(f"Responses: {responses}")

            # sentiment analysis
            # convert responses to list of strings
            if len(responses) == 0:
                print("No responses to analyze for sentiments")
            else:
                responses = responses.tolist()
                # sentiments = sentiment_analysis(responses)
                # print(sentiments)
                
            # summarize the comments
            SUMMARIZER_LOWER_LIMIT = 200
            
            # remove newline from within each individual response/comment
            responses = [response.replace('\n', ' ') for response in responses]

            # concatenate the comments with a newline
            comments = '|'.join(responses)

            len_comments = len(comments)
            print(f"Length of comments: {len_comments}")
            
            model = ["llama3",""]
            summary_text = ""
            if len_comments < SUMMARIZER_LOWER_LIMIT:
                print("Not enough comments to summarize")
                summary = comments
            else:
                print("="*88)
                print(f"len_comments: {len_comments}")
                # print(comments)
                # model = "allenai/led-base-16384"
                # model = "google-t5/t5-small"
                summary_text = get_summary(model, comments)
                # summary = summarizer(model, comments)
                # summary_text = summary[0]['summary_text']
                # print("-"*88)
                # print(f"len_summary: {len(summary_text)}")
                # print(summary_text)

            # write to a file
            # create a file in data/out folder
            # file name has to be subject_catalog_section_instructor_qNo.txt
            # write the question, responses, sentiments, summary
            # create folder for model if it doesn't exist
            # replace '/' in model with - to avoid creating subfolders
            model = model.replace('/', '-')
            _DATA_OUT_MODEL_ = f"{_DATA_OUT_}/{model}"
            if not os.path.exists(f"{_DATA_OUT_MODEL_}"):
                os.makedirs(f"{_DATA_OUT_MODEL_}")
            file_name = f"{_DATA_OUT_MODEL_}/{subject}_{catalog}_{section_nunber}_{instructor}_{column_name}.txt"
            with open(file_name, 'w') as file:
                file.write(f"Question: {question}\n\n")
                # file.write(f"Responses: {len(responses)}\n")
                # file.write(f"Comments: {comments}\n")
                # file.write(f"Sentiments: {sentiments}\n")
                file.write(f"Summary:\n {summary_text}\n\n")
            # close the file
            
#------------------------------------------------------------------------
def process_evals(df_questions, df_responses, _DATA_OUT_):

    # replace instructor name with 'default instructor' if it is Nan
    df_responses['InstructorName'] = df_responses['InstructorName'].fillna('Instructor, Dummy')

    # we get the list of distinct sections (classNbr)
    # then for each ClassNbr we filter out its responses 
    distinct_sections = df_responses['ClassNbr'].unique()
    print(f"Distinct sections: {len(distinct_sections)}")

    for section_nunber in distinct_sections:
        if section_nunber == "10989A":

            # InstructorName,Subject,Catalog,ClassNbr,Term,SessionCode,distanceed,EvaluationForm,Comment1,Column2,Comment2,Column3,Comment3,Column4,Comment4,Column5,Comment5,Column6,Comment6,Column7,Comment7,Column8,Comment8,Column9,Comment9,Column10,Comment10,Column11,Comment11,,
            df_section_responses = df_responses.loc[df_responses['ClassNbr'] == section_nunber]

            # get the instructor, subject, catalog
            instructor = df_section_responses['InstructorName'].iloc[0]
            subject = df_section_responses['Subject'].iloc[0]
            catalog = df_section_responses['Catalog'].iloc[0]

            # instructor name has a comma, so we need to replace it with _
            if len(instructor) > 0 and ',' in instructor:
                instructor = instructor.replace(',', '_')

            print(f"Now processing section: {section_nunber}")
            process_section(instructor, subject, catalog, section_nunber, df_section_responses, df_questions, _DATA_OUT_)
            # processed = True
            print(f"Finished processing section: {section_nunber}")
            print("="*88)

    print("Finished processing all sections")
    print("="*88)

#------------------------------------------------------------------------
def main():
    # Config variables
    _DATA_BASE_DIR_ = r"C:\Users\mmavurap\Desktop\Gitlab Codes\CommentSummarization\src"
    _DATA_IN_ = f"{_DATA_BASE_DIR_}/in"
    _DATA_OUT_ = f"{_DATA_BASE_DIR_}/out"
    _DATA_FILE_QUESTIONS_ = f"{_DATA_IN_}/questions.csv"
    _DATA_FILE_RESPONSES_ = f"{_DATA_IN_}/data.csv"

 
    df_questions, df_responses = load_data_files(_DATA_FILE_QUESTIONS_, _DATA_FILE_RESPONSES_)
    # print_eval_questions_stats(df_questions)
    # print_eval_responses_stats(df_responses)
    # print_df_eval_response_forms(df_questions, df_responses)

    process_evals(df_questions, df_responses, _DATA_OUT_)

#------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
    
    
   