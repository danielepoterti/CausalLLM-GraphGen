from core.utils import draw_graph
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from dowhy.causal_refuters.graph_refuter import GraphRefuter
from itertools import combinations
from dowhy import CausalModel
import re
import pprint
import networkx as nx
import os

def check_causality_LLM(u, v, descriptions, model, domain):
  
    single_template = """You are a helpful assistant to a {domain} expert.\nWhich of the following causal relationship is correct?\n\nA. Changing {a_node} ({a_description}) can cause a change in {b_node} ({b_description}).\nB. Changing {b_node} ({b_description}) can cause a change in {a_node} ({a_description}).\nC. None of the above. No causal relationship exists.\n\nLet's think step-by-step to make sure that we have the right answer.\n\nThen provide your final answer within the tags, <Answer>A/B/C</Answer>."""
    
    prompt_single = PromptTemplate(template= single_template, 
                                   input_variables=["domain", "a_node", "a_description", "b_node", "b_description"]
    )

    llm_chain = LLMChain(prompt=prompt_single, llm=ChatOpenAI(temperature=0, model_name=model))
  # Predicts the causal relationship between the nodes 'u' and 'v' in the domain "credit lending in Germany"
    res = llm_chain.predict(domain=domain, a_node= u, b_node= v, a_description=descriptions[u], b_description = descriptions[v])

  # Prints a formatted message with the domain, node, and their description information
    print(prompt_single.format(domain=domain, a_node= u, b_node= v, a_description=descriptions[u], b_description = descriptions[v]))

  # Prints the prediction result
    print(res)

  # Searches for a specific pattern "<Answer>([A-Z])</Answer>" in the result
    match = re.search(r"<Answer>([A-Z])</Answer>", res)

  # If the pattern is found, returns the letter found else returns None
    if match:
        return match.group(1), res
    else:
        return None
    
def get_violated_CI(model, k):
  # Instantiate GraphRefuter with the model's data
  refuter = GraphRefuter(data=model._data)

  # Get all the nodes from the model's graph
  all_nodes = list(model._graph.get_all_nodes(include_unobserved=False))
  # Count the total number of nodes
  num_nodes = len(all_nodes)

  # Create a list of indices based on the number of nodes
  array_indices = list(range(0, num_nodes))

  # Generate all possible pairs of indices (this is used to test each pair of nodes for conditional independence)
  all_possible_combinations = list(combinations(array_indices, 2))

  # Prepare an empty list to store found conditional independences
  conditional_independences = []

  # For each pair of nodes...
  for combination in all_possible_combinations:
    i = combination[0]
    j = combination[1]
    a = all_nodes[i]
    b = all_nodes[j]

    # Get all nodes excluding 'i' and 'j' (these will be the conditioning set for testing conditional independence)
    if i < j:
        temp_arr = all_nodes[:i] + all_nodes[i + 1 : j] + all_nodes[j + 1 :]
    else:
        temp_arr = all_nodes[:j] + all_nodes[j + 1 : i] + all_nodes[i + 1 :]

    # Generate all combinations of 'k' nodes from temp_arr (this is for testing conditional independence given 'k' nodes)
    k_sized_lists = list(combinations(temp_arr, k))

    # For each combination of 'k' nodes...
    for k_list in k_sized_lists:
        # If 'a' and 'b' are conditionally independent given 'k_list'...
        if model._graph.check_dseparation([str(a)], [str(b)], k_list) == True:
            # Append the found conditional independence to the list
            conditional_independences.append([a, b, k_list])

  # Prepare a list of constraints that need to be checked
  independence_constraints = conditional_independences

  # Refute the model based on these constraints
  refuter.refute_model(independence_constraints=independence_constraints)

  # Filter out the constraints that have been validated by the model
  res = [element for element in independence_constraints if element not in refuter._true_implications]

  # Return the constraints that have been violated (not validated by the model)
  return res

def graph_independence_analysis(G, df, descriptions, modelLLM, domain, classification_variable, file_path):
    
    dir_name = "temp"
    # Percorso completo della cartella
    percorso_cartella = os.path.join(os.getcwd(), dir_name)

    # Verifica se la cartella esiste già
    if os.path.exists(percorso_cartella) == False:
      os.mkdir(percorso_cartella)
    
    nx.drawing.nx_agraph.write_dot(G,"temp/graph.dot")

    column_names = df.columns.tolist()
    column_names = [col for col in column_names if col != classification_variable]

    model=CausalModel(
        data = df,
        treatment= column_names,
        outcome= classification_variable,
        graph= "temp/graph.dot"
        )

    k = 1
    # As long as k is smaller than the number of nodes in the graph minus 1
    while k < len(G.nodes) - 1:
        # Print a separator
        print("-"*10)
        # Print the current value of k
        print("k: " + str(k))

        # Refute the graph with current k value
        res = model.refute_graph(k)

        # Print the refutation result
        print(res)

    # If the refutation result is False, it means the model doesn't satisfy the conditional independencies
        if res.refutation_result == False:
            print("Independencies not satisfied by data:")
            # Get the violated conditional independencies
            violations = get_violated_CI(model, k)
            # Pretty print the violated conditional independencies
            pprint.pprint(violations)

            # Remove the last element from each inner list in violations
            for inner_list in violations:
                del inner_list[-1]

        # Prepare an empty list to store unique conditional independencies
            unique_ci = []
            seen = set()

        # Remove duplicates from violations
            for ci in violations:
                # Convert inner list to a tuple to make it hashable
                ci_tuple = tuple(ci)
                # If this tuple has not been seen before, add it to the unique list
                if ci_tuple not in seen:
                    unique_ci.append(ci)
                    seen.add(ci_tuple)

        # Check if violations are causal via LLM
            for ci in unique_ci:
              # Check causality between the nodes of each unique conditional independence
              resllm, res = check_causality_LLM(ci[0], ci[1], descriptions, modelLLM, domain)

              # Print the conditional independence and the causality result
              print(ci)
              print(resllm)

              if resllm == "A" or resllm == "B":
                single_template = """You are a helpful assistant to a {domain} expert.\nWhich of the following causal relationship is correct?\n\nA. Changing {a_node} ({a_description}) can cause a change in {b_node} ({b_description}).\nB. Changing {b_node} ({b_description}) can cause a change in {a_node} ({a_description}).\nC. None of the above. No causal relationship exists.\n\nLet's think step-by-step to make sure that we have the right answer.\n\nThen provide your final answer within the tags, <Answer>A/B/C</Answer>."""
    
                prompt_single = PromptTemplate(template= single_template, 
                                   input_variables=["domain", "a_node", "a_description", "b_node", "b_description"]
    )
                try:
                    with open(file_path, 'a') as file:
                        text_to_append = "U node: " + ci[0]
                        file.write(text_to_append + "\n")
                        text_to_append = "V node: " + ci[1]
                        file.write(text_to_append + "\n")
                        text_to_append = prompt_single.format(domain=domain, 
                                                              a_node= ci[0], 
                                                              b_node= ci[1], 
                                                              a_description=descriptions[ci[0]], 
                                                              b_description = descriptions[ci[1]])
                        file.write(text_to_append + "\n")
                        text_to_append = "LLM RESPONSE:"
                        file.write(text_to_append + "\n")
                        text_to_append = res
                        file.write(text_to_append + "\n")
                        text_to_append = "-"*10
                        file.write(text_to_append + "\n")
                        
                except IOError:
                    print(f"An error occurred while appending text to the file: {file_path}")
                if resllm == "A":
                  G.add_edge(ci[0], ci[1])
                else:
                  G.add_edge(ci[1], ci[0])

                nx.drawing.nx_agraph.write_dot(G,"temp/graph.dot")
                

                model=CausalModel(
                data = df,
                treatment = column_names,
                outcome = classification_variable,
                graph= "temp/graph.dot"
                )

                k = 1

            # Increment k
            k += 1
        else:
        # If the refutation result is True, just increment k
            k += 1
    
    # Ensure the classification_variable node doesn't have any outgoing edge.
    # If it has, delete them.
    for edge in list(G.out_edges(classification_variable)):
        G.remove_edge(*edge)

    # Check if G has one connected component.
    if nx.number_connected_components(G.to_undirected()) > 1:
        # If not, for each connected component where classification_variable node is not present, 
        # make edges from each node to the classification_variable node.
        for component in nx.connected_components(G.to_undirected()):
            if classification_variable not in component:
                for node in component:
                    G.add_edge(node, classification_variable)
                    try:
                        with open(file_path, 'a') as file:
                            text_to_append = "U node: " + node
                            file.write(text_to_append + "\n")
                            text_to_append = "V node: " + classification_variable
                            file.write(text_to_append + "\n")
                            text_to_append = f"The relationship {node} ---> {classification_variable} has been forced since {node} was not in the same connected component of {classification_variable}"
                            file.write(text_to_append + "\n")
                        
                        
                    except IOError:
                        print(f"An error occurred while appending text to the file: {file_path}")

    return G

def graphs_independence_analysis(result, df, descriptions, domain, classification_variable, result_dir, modelLLM = "gpt-3.5-turbo", ):
    direct_graphs = {}
    for method, G in result.items():
        if isinstance(G, nx.Graph):
            try:
                file_path = result_dir+f"/{method}_explainations.txt"  # Replace with the desired file path and name
                direct_graph = graph_independence_analysis(G, df, descriptions, modelLLM, domain, classification_variable, file_path)
                direct_graphs[method] = direct_graph
            except Exception as e:
                print(f"Error in graph_independence_analysis h for method {method}: {str(e)}")
        else:
            print(f"Error: the value for method {method} is not a Graph: {G}")
    return direct_graphs

def refute_estimate_pipe(model, identified_estimand, estimate, file_path):
    with open(file_path, 'a') as file:
        text_to_append = "*"*10+" REFUTE ESTIMATE "+"*"*10
        file.write(text_to_append + "\n")
    print("*"*10+" REFUTE ESTIMATE "+"*"*10)
    try:
        res_random=model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
        with open(file_path, 'a', encoding='utf-8') as file:
            text_to_append =  str(res_random)
            file.write(text_to_append + "\n")
        print(res_random)
    except Exception as e:
        print(str(e))
        pass

    try:
        res_placebo=model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute")
        with open(file_path, 'a', encoding='utf-8') as file:
            text_to_append = str(res_placebo)
            file.write(text_to_append + "\n")
        print(res_placebo)
    except Exception as e:
        print(str(e))
        pass

    try:
        res_subset = model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter", subset_fraction=0.9)
        with open(file_path, 'a', encoding='utf-8') as file:
            text_to_append =  str(res_subset)
            file.write(text_to_append + "\n")
        print(res_subset)
    except Exception as e:
        print(str(e))
        pass
    
def estimate_and_refute(model, identified_estimand, method_name, file_path):
    try:
        estimate = model.estimate_effect(identified_estimand,
                                         method_name=method_name,
                                         test_significance=True)
        
        try:
            with open(file_path, 'a', encoding='utf-8') as file:
                text_to_append = "*"*15+f" {method_name} "+"*"*15
                file.write(text_to_append + "\n")
                text_to_append = estimate
                file.write(str(text_to_append))
                file.write("\n")
        except IOError:
            print(f"An error occurred while appending text to the file: {file_path}")

        print("*"*15+f" {method_name} "+"*"*15)
        print(estimate)

        refute_estimate_pipe(model, identified_estimand, estimate, file_path)
    except Exception as e:
        print(f"An error occurred while running the {method_name} estimation and refutation: {e}")

def explore_parents(node, graph, df, file_path, visited=None):
    if visited is None:
        visited = set()
    if node in visited:
        return #CASO BASE 1
    
    visited.add(node)
    # Se il nodo non ha padri, è un nodo root
    if not graph.predecessors(node):
        #print(f"{node} è un nodo root")
        return #CASO BASE 2
    # Altrimenti, esplora i padri del nodo
    for parent in graph.predecessors(node):
        with open(file_path, 'a') as file:
            text_to_append = "*"*90
            file.write(text_to_append + "\n")
            text_to_append = f"{parent} ---> {node}"
            file.write(text_to_append + "\n")
            text_to_append = "*"*90
            file.write(text_to_append + "\n")
        print("*"*90)
        print(f"{parent} ---> {node}")
        print("*"*90)

        dir_name = "temp"
        # Percorso completo della cartella
        percorso_cartella = os.path.join(os.getcwd(), dir_name)

        if os.path.exists(percorso_cartella) == False:
          os.mkdir(percorso_cartella)
    
        nx.drawing.nx_agraph.write_dot(graph,"temp/graph.dot")

        #PASSO 1
        model = CausalModel(data = df,
                            treatment= [parent],
                            outcome= [node],
                            graph= "temp/graph.dot"
                            )
        #PASSO 2
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=False)

        if(identified_estimand.no_directed_path is False):
          if(identified_estimand.estimands["backdoor"] is not None):
            #PASSO 3 e 4
            estimate_and_refute(model, identified_estimand, "backdoor.linear_regression", file_path)

          if(identified_estimand.estimands["iv"] is not None):
            #PASSO 3 e 4
            estimate_and_refute(model, identified_estimand, "iv.linear_regression", file_path)

          if(identified_estimand.estimands["frontdoor"] is not None):
            #PASSO 3 e 4
            estimate_and_refute(model, identified_estimand, "frontdoor.linear_regression", file_path)
        else:
          print("Causal relationship does not exists")

        explore_parents(parent, graph, df, file_path, visited)

def explore_parents_graphs(start, result, df, result_dir):

    for method, G in result.items():
        if isinstance(G, nx.Graph):
            try:
                file_path = result_dir+f"/{method}_causalreport.txt"  # Replace with the desired file path and name
                explore_parents(start, G, df, file_path)
            except Exception as e:
                print(f"Error in explore_parents h for method {method}: {str(e)}")
        else:
            print(f"Error: the value for method {method} is not a Graph: {G}")