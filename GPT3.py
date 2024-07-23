import warnings

import bert_score
import ollama
import pandas as pd
import torch
from openai import OpenAI
from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, GPT2Tokenizer

warnings.filterwarnings("ignore")


def load_data(path):
    temp_df_responses = pd.read_csv(path + r'\data.csv', header=0, encoding='cp1252')
    temp_df_questions = pd.read_csv(path + r'\questions.csv', header=0, encoding='cp1252')
    print(temp_df_responses.info())
    print(temp_df_questions.info())

    return temp_df_responses, temp_df_questions


def gpt3_summarize(text, question, max_tokens, temperature=0.7):
    client = OpenAI(api_key="my-openai-apikey")
    response = client.completions.create(
        model="gpt-3.5-turbo",
        prompt=[{"role": "user",
                 "content": f"Summarize the following comments for the following question asked{question}:\n\n{text}\n\nSummary:"}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    summary = response.choices[0].text.strip()
    return summary


def gpt2_summarize(text, question, model_name='gpt2', max_tokens=200, num_beams=5):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    inputs = tokenizer.encode(
        f"Summarize the comments:\n\n{text}",
        return_tensors='pt', max_length=512, truncation=True)

    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    summary_ids = model.generate(
        inputs,
        max_length=max_tokens,
        num_beams=num_beams,
        early_stopping=False,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def ollama_summarize(text,question,model):
    prompt = "Summarize the comments"
    prompt_comments = prompt + text
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt_comments,
        },
    ])

    return response['message']['content']

def gptneo_summarize(text,question,model, tokenizer):
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors='pt',
        max_length=len(text.split()),
        truncation=True
    ).to(device)

    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    summary_ids = model.generate(
        inputs,
        max_length=1000,
        num_beams=5,
        length_penalty=2.0,
        early_stopping=True,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    filePath = r'C:\Users\mmavurap\JupyterNotebbok\Summarization\data'
    df_responses, df_questions = load_data(filePath)

    df_responses_sub = df_responses[df_responses["ClassNbr"] == "13749"]
    df_questions_sub = df_questions[df_questions["FormNumber"] == max(df_responses_sub["EvaluationForm"])]

    models = ["GPT3", "GPT2", "llama3"]
    commentColumn = 1
    for i, df in df_questions_sub.iterrows():
        text = " ".join(df_responses_sub["Comment" + str(commentColumn)].dropna().astype(str))
        question = df["Question"]
        commentColumn += 1
        # gpt3
        summary = gpt3_summarize(text, question, len(text.split()))
        print(summary)
        P, R, F1 = bert_score.score(summary, text, lang="en", verbose=True)
        print(f"Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}")

        # gpt2
        summary = gpt2_summarize(text, question, max_tokens=len(text.split()))
        print(summary)
        P, R, F1 = bert_score.score([summary], [text], lang="en", verbose=True)
        print(f"Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}")

        #ollama
        summary = ollama_summarize(text, question, "llama3")
        print(summary)
        P, R, F1 = bert_score.score([summary], [text], lang="en", verbose=True)
        print(f"Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}")

        # gpt neo
        model_name = "EleutherAI/gpt-neo-1.3B"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        summary = gptneo_summarize(text, question, model, tokenizer)
        print(summary)
        P, R, F1 = bert_score.score([summary], [text], lang="en", verbose=True)
        print(f"Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}")