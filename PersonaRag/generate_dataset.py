
import progressbar
from datasets import load_dataset

# load dataset
wiki = load_dataset("wikimedia/wikipedia", "20231101.en")

docs = {}

num = 0
file_id = f"corpus{str(num)}.tsv"
fp = open(file_id, "w")

print(len(wiki["train"]))


bar = progressbar.ProgressBar(len(wiki["train"])).start()

for i, row in enumerate(wiki["train"]):

    if i % 1000000 == 0 and i > 0:
        fp.close()
        num += 1
        file_id = f"corpus{str(num)}.tsv"
        print(f"open file named {file_id}")
        fp = open(file_id, "w")

    text = row["text"].replace("\n", " ").replace( "\t"," ")
    title = row["title"].replace("\n", " ").replace( "\t"," ")
    fp.write(f"{i}\t{title}\t{text}\n")
    
    if i == len(wiki["train"]):
        break
    bar.update(i+1)
fp.close()


