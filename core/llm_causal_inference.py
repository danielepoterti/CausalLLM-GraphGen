import os
import networkx as nx
import matplotlib.pyplot as plt
import re
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

def check_nodes_in_descriptions(graph, descriptions):
    for node in graph.nodes():
        if node not in descriptions:
            print(f"The node '{node}' does not have a corresponding key in the descriptions dictionary.")
            raise ValueError("Missing description for node.")
    return True

def get_direct_graph(skeleton, descriptions, domain, file_path, model="gpt-3.5-turbo"):
    # check if domain is None or empty
    if domain is None or domain.strip() == '':
        raise ValueError("Error: Domain is not provided or empty.")
        
    if not check_nodes_in_descriptions(skeleton, descriptions):
        raise ValueError("Error: Missing description for one or more nodes.")
    
    single_template = """You are a helpful assistant to a {domain} expert.\nWhich of the following causal relationship is correct?\nA. Changing {a_node} ({a_description}) can cause a change in {b_node} ({b_description}).\nB. Changing {b_node} ({b_description}) can cause a change in {a_node} ({a_description}).\nC. None of the above. No causal relationship exists.\n\nLet's think step-by-step to make sure that we have the right answer.\n\nThen provide your final answer within the tags, <Answer>A/B/C</Answer>."""

    prompt_single = PromptTemplate(template= single_template, 
                                   input_variables=["domain", "a_node", "a_description", "b_node", "b_description"]
    )

    llm_chain = LLMChain(prompt=prompt_single, llm=ChatOpenAI(temperature=0, model_name=model))

    graph = nx.DiGraph()

    for node in skeleton.nodes():
        graph.add_node(node)

    for u, v in skeleton.edges():
        print("U node: " + u)
        print("V node: " + v)
        res = llm_chain.predict(domain=domain, a_node= u, b_node= v, a_description=descriptions[u], b_description = descriptions[v])
        print(prompt_single.format(domain=domain, a_node= u, b_node= v, a_description=descriptions[u], b_description = descriptions[v]))
        print(res)

        match = re.search(r"<Answer>([A-Z])</Answer>", res)

        if match:
            lettera = match.group(1)
            if lettera == "A":
                graph.add_edge(u,v)

                try:
                    with open(file_path, 'a') as file:
                        text_to_append = "U node: " + u
                        file.write(text_to_append + "\n")
                        text_to_append = "V node: " + v
                        file.write(text_to_append + "\n")
                        text_to_append = prompt_single.format(domain=domain, 
                                                              a_node= u, 
                                                              b_node= v, 
                                                              a_description=descriptions[u], 
                                                              b_description = descriptions[v])
                        file.write(text_to_append + "\n")
                        text_to_append = "LLM RESPONSE:"
                        file.write(text_to_append + "\n")
                        text_to_append = res
                        file.write(text_to_append + "\n")
                        text_to_append = "-"*10
                        file.write(text_to_append + "\n")
                        
                except IOError:
                    print(f"An error occurred while appending text to the file: {file_path}")

            elif lettera == "B":
                graph.add_edge(v,u)

                try:
                    with open(file_path, 'a') as file:
                        text_to_append = "U node: " + u
                        file.write(text_to_append + "\n")
                        text_to_append = "V node: " + v
                        file.write(text_to_append + "\n")
                        text_to_append = prompt_single.format(domain=domain, 
                                                              a_node= u, 
                                                              b_node= v, 
                                                              a_description=descriptions[u], 
                                                              b_description = descriptions[v])
                        file.write(text_to_append + "\n")
                        text_to_append = "LLM RESPONSE:"
                        file.write(text_to_append + "\n")
                        text_to_append = res
                        file.write(text_to_append + "\n")
                        text_to_append = "-"*10
                        file.write(text_to_append + "\n")
                        
                except IOError:
                    print(f"An error occurred while appending text to the file: {file_path}")

        print("-"*10)
    
    return graph

def get_all_direct_graphs(result, descriptions, domain, result_dir, model = "gpt-3.5-turbo"):
    direct_graphs = {}
    for method, G in result.items():
        if isinstance(G, nx.Graph):
            file_path = result_dir+f"/{method}_explainations.txt"  # Replace with the desired file path and name
            try:
                with open(file_path, 'w') as file:
                    pass  # The 'pass' statement creates an empty block, so the file remains empty
            except IOError:
                print(f"An error occurred while creating the file: {file_path}")
            try:
                direct_graph = get_direct_graph(G, descriptions, domain, file_path, model)
                direct_graphs[method] = direct_graph
            except ValueError as e:
                print(f"Error in get_direct_graph for method {method}: {str(e)}")
        else:
            print(f"Error: the value for method {method} is not a Graph: {G}")
    return direct_graphs
