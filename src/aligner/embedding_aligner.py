import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from aligner.bert_embedding import BertEmbedding
from utils.data_preprocessor import Preprocessor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


class EmbeddingBasedAligner:
    def __init__(self,
                 template_df: DataFrame,
                 source_df: DataFrame) -> None:
        super().__init__()
        # read only first portion of the data
        self.template_df: DataFrame = template_df
        self.source_df: DataFrame = source_df

        # "bert-base-uncased" performs poor on numbers
        self.embedder: BertEmbedding = BertEmbedding("ramybaly/ner_nerd_fine")

    def _get_cluster_centroid(self, sample: Series):
        # convert each row to string -> get embeddings -> compute centroid
        sample = sample.astype(str)
        embedding_vectors = sample.apply(self.embedder.get_embedding)
        centroid = np.mean(embedding_vectors)
        return centroid

    def _compute_column_to_embedding_map(self, df: DataFrame) -> dict[str, np.ndarray]:
        col_to_centroid: dict[str, np.ndarray] = {}
        for col in df.columns:
            sub_sample: Series = df[col]
            centroid: np.ndarray = self._get_cluster_centroid(sub_sample)
            col_to_centroid[col] = centroid

        return col_to_centroid

    @classmethod
    def _get_alignment(cls, map1: dict[str, np.ndarray], map2: dict[str, np.ndarray]) -> dict[str, str]:
        # Calculate cosine similarity between all pairs of vectors and create a similarity matrix
        similarity_matrix = np.zeros((len(map1), len(map2)))
        keys_dict1 = list(map1.keys())
        keys_dict2 = list(map2.keys())

        for i, key1 in enumerate(keys_dict1):
            for j, key2 in enumerate(keys_dict2):
                similarity_matrix[i, j] = 1 - cosine(map1[key1], map2[key2])

        # print(pd.DataFrame(similarity_matrix, index=keys_dict1, columns=keys_dict2))
        # Hungarian algo for alignment
        alignment_mapping: dict[str, str] = {}
        try:
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            for i, j in zip(row_ind, col_ind):
                alignment_mapping[keys_dict1[i]] = keys_dict2[j]
            # print(alignment_mapping)
        except ValueError:
            print("No alignment found")

        return alignment_mapping

    def get_alignment(self) -> dict[str, str]:
        template_map: dict[str, np.ndarray] = self._compute_column_to_embedding_map(self.template_df)
        source_map: dict[str, np.ndarray] = self._compute_column_to_embedding_map(self.source_df)
        return self._get_alignment(template_map, source_map)


if __name__ == "__main__":
    template_csv_path: str = "/home/users/andreya1/Documents/PERSONAL/llmTestTask/llmTableSchemaMapping/task/template.csv"
    # source_csv_path: str = "/home/users/andreya1/Documents/PERSONAL/llmTestTask/llmTableSchemaMapping/task/table_A.csv"
    source_csv_path: str = "/home/users/andreya1/Documents/PERSONAL/llmTestTask/llmTableSchemaMapping/task/table_B.csv"
    tdf, sdf = Preprocessor.load_data(template_csv_path, source_csv_path=source_csv_path)
    naive_iteration: EmbeddingBasedAligner = EmbeddingBasedAligner(sdf, tdf)
    alignment_mapping: dict[str, str] = naive_iteration.get_alignment()
    print(alignment_mapping)
