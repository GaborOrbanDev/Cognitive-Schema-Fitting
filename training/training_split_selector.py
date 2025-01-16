import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass

data = pd.read_csv("./training/mmlu_all_w_id.csv", sep=";")
subjects = pd.read_csv("./training/mmlu_topics_size.csv", sep=";")


@dataclass
class SubjectSamples:
    subject: str
    ids: list[int]


subject_samples: list[SubjectSamples] = []

sample_indexes: list[int] = []


for subject in subjects.values:
    subject_name = subject[0]
    sample = data.query(f"subject == '{subject_name}'").sample(frac=0.1, replace=False)
    sample_ids = sample["task_id"].values.tolist()
    subject_samples.append(SubjectSamples(subject_name, sample_ids))

for sample in subject_samples:
    sample_indexes.extend(sample.ids)

with open("training/sample_indexes.pkl", "wb") as f:
    pickle.dump(sample_indexes, f)

print(f"Number of samples: {len(sample_indexes)}")
    
    