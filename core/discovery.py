import os
from core.build_causal_skeleton import build_causal_skeleton
from core.causal_graphs_tests import explore_parents_graphs, graphs_independence_analysis
from core.llm_causal_inference import get_all_direct_graphs
from core.utils import draw_all_graphs


def get_graphs(df, descriptions, domain, classification_variable, results_dir,  model = "gpt-3.5-turbo"):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        for file_name in files:
            file_path = os.path.join(results_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    result  = build_causal_skeleton(df)

    draw_all_graphs(result, results_dir, "skeleton")

    result = get_all_direct_graphs(result, descriptions, domain=domain, result_dir=results_dir, model = model)
    result = graphs_independence_analysis(result, 
                                          df, 
                                          descriptions, 
                                          modelLLM = "gpt-3.5-turbo", 
                                          domain = domain, 
                                          classification_variable = classification_variable,
                                          result_dir = results_dir)
    
    draw_all_graphs(result, results_dir, "directed")

    explore_parents_graphs(classification_variable, result, df, results_dir)
    return result