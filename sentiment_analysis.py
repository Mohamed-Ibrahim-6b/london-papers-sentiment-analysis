# import collections
import os
import re
import sys
import traceback

import docx2txt
import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL, clean_up_tokenization_spaces=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def get_sentiment_analysis(text):
    encoded_text = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        "neg": scores[0],
        "neu": scores[1],
        "pos": scores[2],
    }


def get_file_data(text: str, run_sa=True):
    stripped = re.sub(" +", " ", text)
    stripped = re.sub(" \n", "", stripped)
    stripped = re.sub("\xa0", " ", stripped)

    body = stripped.split("Body\n")[1]
    body = body.strip()
    body = body.split("\n\n\n")[0]

    sentiment_analysis = {}
    if run_sa:
        sentiment_analysis = get_sentiment_analysis(body)

    stripped = re.sub("\n+", "\n", stripped)
    text_lines = stripped.split("\n")

    return {
        "body": body,
        "title": text_lines[1],
        "paper": text_lines[3],
        "date": text_lines[4],
        **sentiment_analysis,
    }


def save_articles_in_csv(run_sa=True, verbose=True, truncate_at=-1):
    dirs = [(dir.lower(), f"articles/{dir}") for dir in os.listdir("articles/")]
    i = 0
    result = []
    for category, dir in dirs:
        files = [f"{dir}/{f}" for f in os.listdir(dir)]

        for file in files:
            if truncate_at >= 0 and truncate_at < i:
                break
            i += 1
            if verbose:
                sys.stdout.write("\r* Analysing article: %i" % i)
                sys.stdout.flush()

            text = ""
            try:
                text: str = docx2txt.process(file)
            except Exception as e:
                if verbose:
                    print(
                        f"\nFailed processing docx\n{e}\n{traceback.format_stack()}\n"
                    )

            if text:
                data = {}
                try:
                    data = get_file_data(
                        text=text,
                        run_sa=run_sa,
                    )
                    data["category"] = category
                except Exception as e:
                    if verbose:
                        print(
                            f"\nFailed getting file data\n{e}\n{traceback.format_stack()}\n"
                        )
                if data:
                    result.append(data)

    df = pd.DataFrame.from_dict(result)
    df.to_csv("articles.csv")


if __name__ == "__main__":
    save_articles_in_csv(
        run_sa=True,
        verbose=True,
        truncate_at=50,
    )

    data = pd.read_csv("articles.csv")
    data.to_excel("results.xlsx")
