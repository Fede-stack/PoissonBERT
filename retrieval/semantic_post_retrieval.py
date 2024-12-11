from transformers import AutoModel, AutoTokenizer
import torch

warnings.filterwarnings('ignore')

model = AutoModel.from_pretrained(model_choice)  
tokenizer = AutoTokenizer.from_pretrained(model_choice)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

X_ = []  #list of retrieved documents
y_ = []  #list of scores associated to each retrieved individual's item
error_indices = []  

for i, bdi in enumerate(bdi_items):
    print(i)
#     query_embeddings = model.encode(bdi) #for SentenceTransformers
    inputs_query = tokenizer(bdi, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embeddings = model(**inputs_query, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1].squeeze(1)
    query_embeddings = query_embeddings.cpu().numpy()
    results = []

    for j, docs in enumerate(docss):
        inputs_doc = tokenizer(docs, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            doc_embeddings = model(**inputs_doc, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1].squeeze(1)
        doc_embeddings = doc_embeddings.cpu().numpy()
        #embeddings per documenti
#         doc_embeddings = model.encode(docs) #for SentenceTransformers

        documents_retrieved = []
        #for each query, find similar posts
        for query_emb in query_embeddings:
            try:
                embs = np.concatenate((query_emb.reshape(1, -1), doc_embeddings))
                print(embs.shape)
                data = Data(embs)
                ids, kstars = return_kstar(data, embs, initial_id=None, Dthr=6.67, r='opt', n_iter=10)                
                nns = find_single_k_neighs(embs, 0, kstars[0])

                documents_retrieved.append(np.array(docs)[np.array(nns)-1].tolist())
            except ValueError as e:
                if "array must not contain infs or NaNs" in str(e):
                    error_indices.append((i, j))
                else:
                    raise e

        results.append(documents_retrieved)

    inputs_docs = []
    for res in results:
        inputs_docs.append(extract_docs(res))
        
    X_.append(inputs_docs)
    y_.append(data_scores.iloc[:, i].values.tolist())
