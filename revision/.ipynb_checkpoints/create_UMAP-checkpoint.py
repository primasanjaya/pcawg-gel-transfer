import pandas as pd
import umap

data = pd.read_csv('all_features_pcawg.csv',index_col=0)

st = data.iloc[:,-2:]

data = data.iloc[:,:-2]

reducer = umap.UMAP(random_state=42)
reducer.fit(data.values)
embedding = reducer.transform(data.values)
digits_df = pd.DataFrame(embedding, columns=('x', 'y'))

digits_df.to_csv('UMAP_embedding_revision.csv')