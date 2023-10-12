import os

from core.build_causal_skeleton import build_causal_skeleton
from core.causal_graphs_tests import explore_parents_graphs, graphs_independence_analysis
from core.llm_causal_inference import get_all_direct_graphs
from core.utils import draw_all_graphs


def get_graphs(df, descriptions, immutable_features, domain, 
               classification_variable, results_dir, model):
    """
    Fetches and draws graphs based on the given parameters.

    Args:
        df: DataFrame to build the graphs on.
        descriptions: Descriptions of the features.
        immutable_features: Features that should not be changed.
        domain: Domain of the problem.
        classification_variable: Variable used for classification.
        results_dir: Directory to store the result files.
        model: Model used for learning.
        
    Returns:
        The result of the graph analysis.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    files = [f for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f))]
    for file_name in files:
        os.remove(os.path.join(results_dir, file_name))

    skeleton = build_causal_skeleton(df)

    result = {'pc': skeleton["pc"]} #use skeleton for all 4 algorithms

    draw_all_graphs(result, results_dir, "skeleton")

    result = get_all_direct_graphs(
        result, descriptions, immutable_features, domain=domain,
        result_dir=results_dir, classification_node=classification_variable,
        model=model
    )

    result = graphs_independence_analysis(
        result, df, descriptions, immutable_features, model_llm=model,
        domain=domain, classification_variable=classification_variable,
        result_dir=results_dir
    )

    draw_all_graphs(result, results_dir, "directed")

    explore_parents_graphs(classification_variable, result, df, results_dir)

    return result
