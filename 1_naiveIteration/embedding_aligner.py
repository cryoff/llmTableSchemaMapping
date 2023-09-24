import numpy as np
import pandas as pd
from numpy.random import RandomState
from pandas import DataFrame
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from bert_embedding import BertEmbedding

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


class EmbeddingBasedAligner:
    RANDOM_STATE: RandomState = RandomState(42)
    SAMPLE_SIZE: int = 50
    DATE_FORMAT: str = "DATE_FORMAT"
    NUMBER_FORMAT: str = "NUMBER_FORMAT"
    TEXT_FORMAT: str = "TEXT_FORMAT"

    def __init__(self,
                 template_csv_path: str,
                 source_csv_path: str,
                 dest_pdf_path: str) -> None:
        super().__init__()
        self.template_df: DataFrame = pd.read_csv(template_csv_path)
        self.source_df: DataFrame = pd.read_csv(source_csv_path)
        self.dest_pdf_path: str = dest_pdf_path
        # "bert-base-uncased" performs poor on numbers
        self.embedder: BertEmbedding = BertEmbedding("ramybaly/ner_nerd_fine")

    def _get_cluster_centroid(self, sample: DataFrame):
        # convert each row to string
        sample = sample.astype(str)
        # apply function "fun" to each row and save the result in a new column
        # sample["embedding"] = sample.iloc[:, 0].apply(lambda x: self.embedder.get_embedding(x))
        embedding_vectors = sample.apply(self.embedder.get_embedding)
        centroid = np.mean(embedding_vectors)
        return centroid

    def _compute_column_to_embedding_map(self, df: DataFrame) -> dict[str, np.ndarray]:
        sample_size = min(self.SAMPLE_SIZE, len(df))
        col_to_centroid: dict[str, np.ndarray] = {}

        for col in df.columns:
            sub_sample: DataFrame = df[col].sample(
                n=sample_size,
                random_state=self.RANDOM_STATE)
            centroid: np.ndarray = self._get_cluster_centroid(sub_sample)
            col_to_centroid[col] = centroid

        return col_to_centroid

    def _get_alignment(self, template_map: dict[str, np.ndarray], source_map: dict[str, np.ndarray]):
        # Calculate cosine similarity between all pairs of vectors and create a similarity matrix
        similarity_matrix = np.zeros((len(template_map), len(source_map)))
        keys_dict1 = list(template_map.keys())
        keys_dict2 = list(source_map.keys())

        for i, key1 in enumerate(keys_dict1):
            for j, key2 in enumerate(keys_dict2):
                similarity_matrix[i, j] = 1 - cosine(template_map[key1], source_map[key2])

        # print(pd.DataFrame(similarity_matrix, index=keys_dict1, columns=keys_dict2))
        # Hungarian algo for alignment
        try:
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            alignment_mapping = {}
            for i, j in zip(row_ind, col_ind):
                alignment_mapping[keys_dict1[i]] = keys_dict2[j]
            # print(alignment_mapping)
        except ValueError:
            print("No alignment found")

    def process(self):
        if len(self.template_df.columns) < len(self.source_df.columns):
            raise ValueError("Number of columns in template is smaller than in source")

        template_map: dict[str, np.ndarray] = self._compute_column_to_embedding_map(self.template_df)
        source_map: dict[str, np.ndarray] = self._compute_column_to_embedding_map(self.source_df)
        self._get_alignment(template_map, source_map)


if __name__ == "__main__":
    template_csv_path: str = "/home/users/andreya1/Documents/PERSONAL/llmTestTask/llmTableSchemaMapping/task/template.csv"
    # source_csv_path: str = "/home/users/andreya1/Documents/PERSONAL/llmTestTask/llmTableSchemaMapping/task/table_A.csv"
    source_csv_path: str = "/home/users/andreya1/Documents/PERSONAL/llmTestTask/llmTableSchemaMapping/task/table_B.csv"
    dest_pdf_path: str = "path_to_dest_pdf"
    naive_iteration: EmbeddingBasedAligner = EmbeddingBasedAligner(template_csv_path, source_csv_path, dest_pdf_path)
    naive_iteration.process()
