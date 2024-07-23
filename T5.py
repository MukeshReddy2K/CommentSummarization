import pandas as pd
import os
import warnings
import pprint

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

warnings.filterwarnings("ignore")

def summarize_text(text, model, tokenizer, max_length, min_length, num_beams=5):
    inputs = tokenizer.encode(
        "summarize the comments in an elaborate way: " + text,
        return_tensors='pt',
        max_length=len(text.split()),
        truncation=True
    )

    # Generate the summary
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=False,
        min_length=min_length
    )

    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    pp = pprint.PrettyPrinter()

    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")

    filePath = r'C:\Users\mmavurap\Desktop\Gitlab Codes\CommentSummarization\responses.xlsx'

    data = pd.read_excel(io=filePath, sheet_name="Data", header=0)

    question = pd.read_excel(io=filePath, sheet_name="Questions", header=0)

    df = data[data["ClassNbr"] == 13749]

    formQuestions = question[question["FormNumber"] == max(df["EvaluationForm"])]

    # summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    text = " ".join(df["Comment1"].dropna().astype(str))

    print(text)

    print(len(text.split()))
    # print(summarizer(text,max_length= 1000, min_length=30, do_sample=False))
    print("=" * 50)
    max_length = [300, 500, 600, 700]
    min_length = [50, 80, 100, 200]
    numbeans = [3, 5, 8, 10]
    for i in range(4):
        print("max_length, min_length, numbeans = {", max_length[i], min_length[i], numbeans[i], "}")
        summary_text = summarize_text(text, model, tokenizer, max_length[i], min_length[i], numbeans[i])

        pp.pprint(summary_text)

        print("=" * 50)
