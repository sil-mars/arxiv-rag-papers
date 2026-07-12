import json
import pandas as pd

class Loader:

  def __init__(self, p):
    self.path = p

  def clean_data(self, text):
      text = str(text) 
      text = text.replace("\n", " ") 
      text = " ".join(text.split()) # normalize spaces
      return text

  def preprocessing(self, item):
    title = self.clean_data(item["title"])
    abstract = self.clean_data(item["abstract"])
    return title + "\n" + abstract

  def read_data(self):
    data = []
    # read line by line
    with open(self.path, "r") as f:

        for line in f:
            item = json.loads(line)

            text = self.preprocessing(item)

            data.append({
                    "text": text,
                    "title": item["title"],
                    "category": item["categories"],
                    "id": item["id"]
                })

    return data

  def chunk(self, text, words_lim=100): # not used
    words = text.split()
    step = int(words_lim * 0.8)

    res = [
        " ".join(words[i:i+words_lim])
        for i in range(0, len(words), step)
    ]
    return res