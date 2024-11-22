import pandas as pd
import ollama
import Constants as C
import pymssql
from datetime import datetime

# -----------------------------------------------------------------------
def load_summary_into_db(df, cursor):
    insert_sql = "insert into [UH - Faculty Course Evaluations Reports].dbo.temp_tblSummary (term, acadgroup, acadorg," \
                 "classnbr, subject, catalog, dateInserted, summary, qValue, question, modelName, modelDescription ) values (%d,%d,%s,%s,%s,%d,%s," \
                 "%s,%s,%s,%s,%s)"
    data = list(df.itertuples(index=False, name=None))
    cursor.executemany(insert_sql, data)

# ------------------------------------------------------------------------
def get_summary(model, comments, question):
    prompt = "Please provide a concise summary of the following comments. If the comments are few in number, avoid adding " \
             "any extra information or assumptions—focus only on the core ideas presented. Do not introduce any new details " \
             "or speculations. Be precise and factual in summarizing"
    prompt_comments = prompt + comments
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt_comments,
        },
    ])
    return response['message']['content']

# ------------------------------------------------------------------------
def process_section(df_section_responses, df_section_questions, df_summary, evalForm, section_number, term, df_errorlog):
    print("=" * 88)
    print(f"Section: {section_number}")

    try:

        subject = df_section_responses['subject'].iloc[0]
        df_section_responses = df_section_responses.copy()
        df_section_responses['catalog'] = df_section_responses['catalog'].apply(
            lambda x: x[0] if isinstance(x, tuple) else x)
        catalog = df_section_responses['catalog'].iloc[0],
        acadgroup = df_section_responses['acadgroup'].iloc[0]
        acadorg = df_section_responses['acadorg'].iloc[0]

        if evalForm is None:
            print("No form number found")
            return
        else:
            for index, row in df_section_questions.iterrows():
                question = row['Question']
                qvalue = row["qValues"].lower()
                responses = df_section_responses[row["qValues"].lower()]
                responses = responses.dropna()
                responses = responses[responses != '']
                if len(responses) == 0:
                    summary = ''
                else:
                    responses = responses.tolist()
                    summary = ''

                    # remove newline from within each individual response/comment
                    responses = [response.replace('\n', ' ') for response in responses]

                    # concatenate the comments with a newline
                    text = ' '.join(responses)

                    description = 'LLaMA 3 (Large Language Model Meta AI) is a state-of-the-art language model designed ' \
                                  'for natural language processing tasks. It is fine-tuned with 8 billion parameters. ' \
                                  'trained on diverse datasets to follow complex instructions.'

                    print(f"Question: {question}")

                    print("Model start time:", datetime.now())

                    text = get_summary('llama3', text, question)

                    textList = ''

                    print("Model end time:", datetime.now())

                    if '/n/n' in text:
                        textList = text.split('/n/n', 1)
                        if len(textList) == 2:
                            summary = textList[1]
                    else:
                        summary = text
                    print(summary)
                    model = 'LLaMA 3'
                    df_summary.loc[len(df_summary)] = [term, acadgroup, acadorg, section_number, subject, catalog,
                                                       datetime.now(), summary, qvalue, question, model, description]
        print('-' * 50)
    except Exception as e:
        df_errorlog.loc[len(df_errorlog)] = [term, section_number, e]



# ------------------------------------------------------------------------
def process_evals(df_questions, df_responses, term=2190, acadgroup = None):
    # replace instructor name with 'default instructor' if it is Nan
    df_responses['InstructorName'] = df_responses['InstructorName'].fillna('Instructor, Dummy')
    print(f"Process started for {acadgroup} at {datetime.now()}" )
    df_responses["Processed"] = False
    # we get the list of distinct sections (classNbr)
    # then for each ClassNbr we filter out its responses
    distinct_sections = df_responses['classnbr'].unique()
    print(f"Distinct sections: {len(distinct_sections)}")

    df_summary = pd.DataFrame(
        columns=['term', 'Acadgroup', 'acadorg', 'classnbr', 'subject', 'catalog', 'dateInserted',
                 'summary', 'qValue', 'Question', 'ModelName', 'ModelDescription'])
    df_errorlog = pd.DataFrame(columns=["Term","ClassNbr", "Error"])

    for section_number in distinct_sections:
            df_section_responses = df_responses.loc[df_responses['classnbr'] == section_number]

            evalForm = int(df_section_responses["evalForm"].iloc[0])

            df_section_questions = df_questions.loc[df_questions['formnumber'] == evalForm]

            if df_section_responses['distEval'].iloc[0].any():
                df_section_questions = pd.concat([df_section_questions, df_questions[df_questions['formnumber'] == 23]],
                                                 ignore_index=True)

            print(f"Now processing classnbr: {section_number} {datetime.now()}")
            process_section(df_section_responses, df_section_questions, df_summary, evalForm, section_number, term, df_errorlog)
            df_responses.loc[df_responses['classnbr'] == section_number, 'processed'] = True

    print("Finished processing all sections")
    print("=" * 88)

    return df_summary, df_errorlog


# ------------------------------------------------------------------------
def main():
    # defining term, acadgroup
    term = 2180
    acadgroup = [15]

    connection = pymssql.connect(server=C.MEC_SQL_HOST, user=C.MEC_SQL_USERNAME, password=C.MEC_SQL_PASSWORD,
                                 database=C.MEC_SQL_DB_FACULTY_COURSE_EVALUATIONS)
    cursor = connection.cursor(as_dict=True)

    for i in acadgroup:
        sql = "EXEC dbo.get_comments_for_summarization @term = " + str(term) + ", @acadgroup = " + str(i)
        cursor.execute(sql)

        df_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

        cursor.nextset()

        df_questions = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

        summary_dataset, error_dataset = process_evals(df_questions, df_data, term, i)

        summary_dataset.to_csv(f'summary_{i}.csv', index=False)

        if len(error_dataset) > 0:
            error_dataset.to_csv(f'error_log_{i}.csv', inplace = False)

        load_summary_into_db(summary_dataset, cursor)

        connection.commit()

    cursor.close()

    connection.close()


# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
