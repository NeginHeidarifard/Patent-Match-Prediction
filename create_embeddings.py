"""
create embeddings for all patent texts with sentence-transformers, and save the embeddings as a numpy array
"""

import argparse
# import faiss
# from faiss import write_index, read_index
from sentence_transformers import SentenceTransformer, models

import pandas as pd
import numpy as np

import json
import os


def main():
    parser = argparse.ArgumentParser(description='Create embeddings for all patent texts with sentence-transformers, and save the embeddings as a numpy array')
    parser.add_argument('--model', '-m', type=str, default='AI-Growth-Lab/PatentSBERTa', help='The model to use for embeddings')    # all models that support sentence-transformers architecture can be used : https://huggingface.co/models?library=sentence-transformers&sort=downloads
    parser.add_argument('--pooling', '-p', type=str, default='mean', choices=['mean', 'max', 'cls'], help='The pooling strategy to use for embeddings') # see more pooling strategy options here: https://github.com/UKPLab/sentence-transformers/blob/0ab62663b5b1425f7df05aad34636f7eb6e3a07c/sentence_transformers/models/Pooling.py#L9
    parser.add_argument('--input_file', '-i', type=str, help='The input file to create embeddings for', 
                        default='/bigstorage/DATASETS_JSON/Content_JSONs/Citing_2020_Cleaned_Content_12k/Citing_Train_Test/citing_TRAIN.json')
    parser.add_argument('--output_dir', '-o', type=str, default='embeddings_precalculated', help='The output file to save the embeddings to')    
    args = parser.parse_args()


    # Load the input json file
    with open(args.input_file, 'r') as json_file:
        data = json.load(json_file)
    print(f"Loaded {len(data)} documents from {args.input_file}")
    
    # convert the jsonl to a pandas dataframe (long table of content)
    columns = ['Application_Number', 'Application_Date', 'Application_Category', 'Content_Type', 'Content']

    data_accumulator = []
    for doc in data:
        for content_type, content in doc['Content'].items():
            # Create a dictionary for each row and append it to the list
            row_data = {
                'Application_Number': doc['Application_Number'], 
                'Application_Date': doc['Application_Date'], 
                'Application_Category': doc['Application_Category'], 
                'Content_Type': content_type, 
                'Content': content
            }
            data_accumulator.append(row_data)

    # Create the DataFrame from the accumulated data list only once
    df = pd.DataFrame(data_accumulator, columns=columns)
    # print(df.head())
    # print(df.shape) # (2382315, 5)


    # Load the model
    base_model = models.Transformer(args.model, max_seq_length=512)
    pooling_model = models.Pooling(base_model.get_word_embedding_dimension(), pooling_mode=args.pooling)
    model = SentenceTransformer(modules=[base_model, pooling_model]).to('cuda')

    # Encode the corpus with potentially adjusted batch size
    corpus_embeddings = model.encode(df['Content'].tolist(), show_progress_bar=True, batch_size=128)  # Adjust batch size as needed
    print(f"Encoded {len(corpus_embeddings)} documents")

    # Save the embeddings as a numpy array
    if not os.path.exists(args.output_dir): # create the output directory if it doesn't exist
        os.makedirs(args.output_dir)

    np.save(f'{args.output_dir}/embeddings_{args.model.split("/")[-1]}_{args.pooling}.npy', corpus_embeddings)
    print(f"Saved the embeddings to {args.output_dir}/embeddings_{args.model.split('/')[-1]}_{args.pooling}.npy")



if __name__ == '__main__':
    main()








