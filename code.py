import pyterrier as pt
import os
import shutil

if not pt.started():
    pt.init()

dir_path = input("Enter the directory's path that contains the documents to search (or -1 to skip): ")

if dir_path != "-1":
    if not os.path.isdir(dir_path):
        print("Entered path does not point to a directory.")
        exit()

    docs = []

    for i, filename in enumerate(os.listdir(dir_path)):
        print("docid " + str(i) + ": " + dir_path + "/" + filename)

        docs.append(dir_path + "/" + filename)

    if os.path.exists("./temp_index"):
        shutil.rmtree("./temp_index")

    index = pt.FilesIndexer("./temp_index").index(docs)


    batch = pt.BatchRetrieve(index, wmodel="BM25")

    while True:
        query = input("\nEnter search query (or -1 to exit search): ")

        if query == "-1":
            break

        print(batch.search(query))

    if os.path.exists("./temp_index"):
        shutil.rmtree("./temp_index")

dataset = pt.datasets.get_dataset("irds:antique/test")

print("\nExperiments:\nEnglish Tokenizer:")

if not os.path.exists("./indices/eng_token"):
    index = pt.IterDictIndexer("./indices/eng_token", tokeniser="english").index(dataset.get_corpus_iter())
else:
    index = pt.IndexRef.of("./indices/eng_token")

print(pt.Experiment(
    [pt.BatchRetrieve(index, wmodel="Tf"),
     pt.BatchRetrieve(index, wmodel="TF_IDF"),
     pt.BatchRetrieve(index, wmodel="BM25")],
    dataset.get_topics(),
    dataset.get_qrels(),
    ['recall_5', 'recall_10', 'P_5', 'P_10', 'ndcg']))

print("\nUTF Tokenizer:")

if not os.path.exists("./indices/utf_token"):
    index = pt.IterDictIndexer("./indices/utf_token", tokeniser="utf").index(dataset.get_corpus_iter())
else:
    index = pt.IndexRef.of("./indices/utf_token")

print(pt.Experiment(
    [pt.BatchRetrieve(index, wmodel="Tf"),
     pt.BatchRetrieve(index, wmodel="TF_IDF"),
     pt.BatchRetrieve(index, wmodel="BM25")],
    dataset.get_topics(),
    dataset.get_qrels(),
    ['recall_5', 'recall_10', 'P_5', 'P_10', 'ndcg']))

print("\nWith Stemmer:")

if not os.path.exists("./indices/stem"):
    index = pt.IterDictIndexer("./indices/stem", stemmer="porter").index(dataset.get_corpus_iter())
else:
    index = pt.IndexRef.of("./indices/stem")

print(pt.Experiment(
    [pt.BatchRetrieve(index, wmodel="Tf"),
     pt.BatchRetrieve(index, wmodel="TF_IDF"),
     pt.BatchRetrieve(index, wmodel="BM25")],
    dataset.get_topics(),
    dataset.get_qrels(),
    ['recall_5', 'recall_10', 'P_5', 'P_10', 'ndcg']))

print("\nWithout Stemmer:")

if not os.path.exists("./indices/no_stem"):
    index = pt.IterDictIndexer("./indices/no_stem", stemmer="none").index(dataset.get_corpus_iter())
else:
    index = pt.IndexRef.of("./indices/no_stem")

print(pt.Experiment(
    [pt.BatchRetrieve(index, wmodel="Tf"),
     pt.BatchRetrieve(index, wmodel="TF_IDF"),
     pt.BatchRetrieve(index, wmodel="BM25")],
    dataset.get_topics(),
    dataset.get_qrels(),
    ['recall_5', 'recall_10', 'P_5', 'P_10', 'ndcg']))

print("\nWith Stopword Removal:")

if not os.path.exists("./indices/stop"):
    index = pt.IterDictIndexer("./indices/stop", stopwords="terrier").index(dataset.get_corpus_iter())
else:
    index = pt.IndexRef.of("./indices/stop")

print(pt.Experiment(
    [pt.BatchRetrieve(index, wmodel="Tf"),
     pt.BatchRetrieve(index, wmodel="TF_IDF"),
     pt.BatchRetrieve(index, wmodel="BM25")],
    dataset.get_topics(),
    dataset.get_qrels(),
    ['recall_5', 'recall_10', 'P_5', 'P_10', 'ndcg']))

print("\nWithout Stopword Removal:")

if not os.path.exists("./indices/no_stop"):
    index = pt.IterDictIndexer("./indices/no_stop", stopwordws="none").index(dataset.get_corpus_iter())
else:
    index = pt.IndexRef.of("./indices/no_stop")

print(pt.Experiment(
    [pt.BatchRetrieve(index, wmodel="Tf"),
     pt.BatchRetrieve(index, wmodel="TF_IDF"),
     pt.BatchRetrieve(index, wmodel="BM25")],
    dataset.get_topics(),
    dataset.get_qrels(),
    ['recall_5', 'recall_10', 'P_5', 'P_10', 'ndcg']))

