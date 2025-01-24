from langgraph.graph.state import CompiledStateGraph


def schema_selector(prediction: str, schemas: dict[str, CompiledStateGraph]) -> CompiledStateGraph:
    """Selects the cognitive schema based on the prediction result.

    Parameters:
    -----------
    prediction : str
        The predicted cognitive schema label.
    schemas : dict[str, CompiledStateGraph]
        A dictionary of compiled cognitive schemas with schema labels as keys.

    Returns:
    --------
    CompiledStateGraph
        The compiled cognitive schema corresponding to the predicted label.
    """
    return schemas[prediction]