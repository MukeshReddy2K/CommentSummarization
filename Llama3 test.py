import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import pymssql
import Constants as C


# Install dependencies in Databricks (Run this in a separate cell)
# %pip install pyodbc transformers torch pandas

# -----------------------------------------------------------------------
def load_summary_into_db(df, cursor):
    insert_sql = "insert into [UH - Faculty Course Evaluations Reports].dbo.tblSummary (term, acadgroup, acadorg," \
                 "classnbr, subject, catalog, dateInserted, summary, qValue, question, modelName, modelDescription ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    data = list(df.itertuples(index=False, name=None))
    cursor.executemany(insert_sql, data)

# ------------------------------------------------------------------------
def get_summary(model, tokenizer, comments, question):
    prompt = f"Summarize the comments asked for the following question: {question}:"
    prompt_comments = prompt + comments

    # Tokenize input text
    inputs = tokenizer(prompt_comments, return_tensors="pt", truncation=True)

    # Generate summary using LLaMA 3 model
    summary_ids = model.generate(inputs['input_ids'], max_length=1500, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# ------------------------------------------------------------------------
def process_section(df_section_responses, df_section_questions, df_summary, evalForm, section_number, term, model, tokenizer):
    print("=" * 88)
    print(f"Section: {section_number}")
    subject = df_section_responses['subject'].iloc[0]
    df_section_responses['catalog'] = df_section_responses['catalog'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
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

                # Remove newline from within each individual response/comment
                responses = [response.replace('\n', ' ') for response in responses]

                # Concatenate the comments
                text = ' '.join(responses)

                description = 'LLaMA 3 (Large Language Model Meta AI) is a state-of-the-art language model designed ' \
                              'for natural language processing tasks. It is fine-tuned with 8 billion parameters. ' \
                              'trained on diverse datasets to follow complex instructions.'

                print(f"Question: {question}")
                print("Model start time:", datetime.now())

                summary = get_summary(model, tokenizer, text, question)

                print("Model end time:", datetime.now())

                df_summary.loc[len(df_summary)] = [term, acadgroup, acadorg, section_number, subject, catalog,
                                                   datetime.now(), summary, qvalue, question, "LLaMA 3", description]

                print('-' * 50)

# ------------------------------------------------------------------------
def process_evals(df_questions, df_responses, term, model, tokenizer):
    df_responses['InstructorName'] = df_responses['InstructorName'].fillna('Instructor, Dummy')
    df_responses["Processed"] = False

    distinct_sections = df_responses['classnbr'].unique()
    print(f"Distinct sections: {len(distinct_sections)}")

    df_summary = pd.DataFrame(columns=['term', 'Acadgroup', 'acadorg', 'classnbr', 'subject', 'catalog', 'dateInserted',
                                       'summary', 'qValue', 'Question', 'ModelName', 'ModelDescription'])

    for section_number in distinct_sections:
        if section_number == '23572':

            df_section_responses = df_responses.loc[df_responses['classnbr'] == section_number]
            evalForm = int(df_section_responses["evalForm"].iloc[0])
            df_section_questions = df_questions.loc[df_questions['formnumber'] == evalForm]

            if df_section_responses['distEval'].iloc[0].any():
                df_section_questions = pd.concat([df_section_questions, df_questions[df_questions['formnumber'] == 23]],
                                                 ignore_index=True)

            print(f"Now processing classnbr: {section_number} {datetime.now()}")
            process_section(df_section_responses, df_section_questions, df_summary, evalForm, section_number, term, model, tokenizer)
            df_responses.loc[df_responses['classnbr'] == section_number, 'processed'] = True

    print("Finished processing all sections")
    return df_summary

# ------------------------------------------------------------------------
def main():
    term = 2190
    acadgroup = 37
    connection = pymssql.connect(server=C.MEC_SQL_HOST, user=C.MEC_SQL_USERNAME, password=C.MEC_SQL_PASSWORD,
                                 database=C.MEC_SQL_DB_FACULTY_COURSE_EVALUATIONS)
    cursor = connection.cursor(as_dict=True)

    sql = "EXEC dbo.get_comments_for_summarization @term = " + str(term) + ", @acadgroup = " + str(acadgroup)
    cursor.execute(sql)

    df_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    cursor.nextset()

    df_questions = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    # tokenizer = AutoTokenizer.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")
    # model = AutoModelForCausalLM.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")
    # # Load model directly
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    model_name = "meta-llama/LLaMA-3-7b"
    token = "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"  # Replace with your actual token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    # Process evaluations
    summary_dataset = process_evals(df_questions, df_data, term, model, tokenizer)


    # Save summary to CSV
    summary_dataset.to_csv('summary.csv', index=False)

    # Insert the summary data into SQL Server
    load_summary_into_db(summary_dataset, cursor)

    connection.commit()
    cursor.close()
    connection.close()

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
