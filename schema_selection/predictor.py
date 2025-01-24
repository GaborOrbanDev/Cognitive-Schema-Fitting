import pandas as pd
from langchain_community.vectorstores import FAISS


class SchemaPredictor:
    """A class for predicting cognitive schemas based on semantic similarity using a vector store.
    Other prediction methods can also be applied by extending this class.

    Attributes
    ----------
    vectorstore : FAISS
        A FAISS vector store used for semantic similarity searches.
    labels : list
        A list of possible cognitive schema labels. Defaults to ["CoT", "CoT-SC", "CoT-SR", "SPP", "ToT"].

    Methods
    -------
    semantic_similarity_based_predicter(task: str, k: int = 30) -> str
        Predicts the cognitive schema most relevant to the given task using semantic similarity.
    """
    
    def __init__(self, vectorstore: FAISS, labels: list | None = None):
        self.vectorstore = vectorstore

        if labels is None:
            self.labels = ["CoT", "CoT-SC", "CoT-SR", "SPP", "ToT"]
        else:
            self.labels = labels
    

    def semantic_similarity_based_predicter(self, task: str, k: int = 30) -> str:
        """Predicts the cognitive schema most relevant to the given task based on semantic similarity.

        Parameters:
        -----------
        task : str
            The task description for which the cognitive schema is to be predicted.
        k : int, optional
            The number of top relevant documents to retrieve from the vector store. Defaults to 30.

        Returns:
        --------
        str
            The predicted cognitive schema label.
        
        Notes:
        ------
        The method performs a similarity search in the vector store using the provided task as the query. It filters
        results by an accuracy threshold and retrieves metadata from the documents. The cognitive schema label is
        predicted based on the most frequent label in the retrieved documents, with ties broken by inference time
        and total token usage.
        """

        relevant_documents = self.vectorstore.similarity_search(
            query=task,
            k=k,
            filter={"accuracy": 1},
            fetch_k=7000
        )

        data = pd.DataFrame([
            {
                "cognitive_schema": d.metadata["cognitive_schema"],
                "inference_time": d.metadata["inference_time"],
                "total_tokens": d.metadata["total_tokens"]
            }
            for d in relevant_documents
        ])
        
        label: str = data.groupby("cognitive_schema").agg(
            size = pd.NamedAgg(column="cognitive_schema", aggfunc="size"),
            mean_inference_time = pd.NamedAgg(column="inference_time", aggfunc="mean"),
            mean_total_tokens = pd.NamedAgg(column="total_tokens", aggfunc="mean")
        ).sort_values(
            by=["size", "mean_inference_time", "mean_total_tokens"], 
            ascending=[False, True, True]
        ).index[0]

        return label