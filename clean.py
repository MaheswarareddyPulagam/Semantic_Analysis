import re
import os
import numpy as np

def clean_document(text):
    """
    Cleans a single document by removing metadata and unwanted characters.
    """

    # Remove headers (everything before the first blank line)
    parts = text.split("\n\n", 1)
    if len(parts) > 1:
        text = parts[1]

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove strange characters while keeping basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()'\" ]", " ", text)

    return text.strip()


def load_dataset(path):
    """
    Loads and cleans all documents from the dataset directory.
    """

    documents = []
    labels = []

    for category in os.listdir(path):

        category_path = os.path.join(path, category)

        if os.path.isdir(category_path):

            for file in os.listdir(category_path):

                file_path = os.path.join(category_path, file)

                # Ensure it's a valid file
                if not os.path.isfile(file_path):
                    continue

                try:
                    with open(file_path, "r", encoding="latin1") as f:

                        text = f.read()
                        text = clean_document(text)

                        # Skip empty or extremely small documents
                        if len(text) < 20:
                            continue

                        documents.append(text)
                        labels.append(category)

                except Exception:
                    # Skip unreadable files
                    pass

    return documents, labels


# MAIN PROGRAM
if __name__ == "__main__":

    path = "20_newsgroups"

    docs, labels = load_dataset(path)

    print("Total documents:", len(docs))
    print("Example label:", labels[0])

    print("\nExample cleaned document:\n")
    print(docs[0][:500])
    np.save("docs.npy", np.array(docs))