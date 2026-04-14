# Copyright (c) 2026 Emanuele Selleri - University of Parma
# Licensed under the MIT License
#!/usr/bin/env python3
import pandas as pd
from ete3 import Tree, TreeStyle, NodeStyle, faces, AttrFace
import matplotlib.pyplot as plt
import seaborn as sns
import re

GAIN_COST = 2
LOSS_COST = 1

print("=== Script start ===")

tree_file = "core.newick"
print(f"Newick file found: {tree_file}")

with open(tree_file) as f:
    tree_str = f.read().strip()

print(f"Phylogenetic tree lengh: {len(tree_str)}")

format_est = 0
if re.search(r"\)\d+:", tree_str):
    format_est = 1
print(f"Putative tree type: {format_est}")

tree = Tree(tree_file, format=format_est)
print(f" Putative leaf number: {len(tree.get_leaves())}")

matrix_file = "matrix_1_0.csv"
print(f"Matrix file found: {matrix_file}")
matrix = pd.read_csv(matrix_file, header=0, index_col=0).T

print(f"Genomes in matrix: {matrix.shape[0]}")
print(f"Genes in matrix: {matrix.shape[1]}")

leaf_tree = set(leaf.name for leaf in tree.get_leaves())
leaf_matrix = set(matrix.index)

leaf_missing = leaf_tree - leaf_matrix
leaf_extra = leaf_matrix - leaf_tree

print(f"leafs not found in the matrix file: {len(leaf_mancanti)}")
print(f"Extra Genomes in the  matrix file: {len(leaf_extra)}")

with open("leaf_missing.txt", "w") as f:
    for nome in sorted(leaf_mancanti):
        f.write(nome + "\n")

for nome in leaf_missing:
    node = tree.search_nodes(name=nome)
    if node:
        print(f"Remove missing leaf: {nome}")
        node[0].detach()

def wagner_parsimony(tree, states):
    for node in tree.traverse():
        node.add_feature("state", None)

    for leaf in tree.get_leaves():
        leaf.state = states.get(leaf.name, 0)

    def bottom_up(node):
        if node.is_leaf():
            node.costs = {
                0: 0 if node.state == 0 else GAIN_COST,
                1: 0 if node.state == 1 else LOSS_COST
            }
        else:
            for child in node.get_children():
                bottom_up(child)
            node.costs = {}
            for s in [0, 1]:
                cost_sum = 0
                for child in node.get_children():
                    cost_child = min(
                        child.costs[s2] + (GAIN_COST if s2 > s else LOSS_COST if s2 < s else 0)
                        for s2 in [0, 1]
                    )
                    cost_sum += cost_child
                node.costs[s] = cost_sum

    bottom_up(tree)

    def top_down(node, parent_state=None):
        if parent_state is None:
            node.state = min(node.costs, key=node.costs.get)
        else:
            options = [(node.costs[s] + (GAIN_COST if s > parent_state else LOSS_COST if s < parent_state else 0), s) for s in [0, 1]]
            node.state = min(options)[1]
        for child in node.get_children():
            top_down(child, node.state)

    top_down(tree)

    gains = losses = 0
    for node in tree.traverse("postorder"):
        if not node.is_root():
            if node.state > node.up.state:
                gains += 1
                node.add_feature("event", "gain")
            elif node.state < node.up.state:
                losses += 1
                node.add_feature("event", "loss")
            else:
                node.add_feature("event", "none")
    return gains, losses, tree

print("COGs analysis start")

results_wagner = []
genome_events = []
genome_event_counts = {}

genome_parent_event_counts = {}
node_event_counts = {}

for i, fam in enumerate(matrix.columns):
    print(f"Analyses COG {i+1}/{len(matrix.columns)}: {fam}")
    tree_copy = tree.copy()
    states = matrix[fam].to_dict()

    gains, losses, t_w = wagner_parsimony(tree_copy, states)
    results_wagner.append({"family": fam, "gains": gains, "losses": losses})

    for node in t_w.traverse("postorder"):
        if hasattr(node, "event") and node.event in {"gain", "loss"}:
            for leaf in node.get_leaves():
                genome_name = leaf.name
                genome_events.append({
                    "family": fam,
                    "event": node.event,
                    "genome": genome_name
                })
                if genome_name not in genome_event_counts:
                    genome_event_counts[genome_name] = {"gains": 0, "losses": 0}
                if node.event == "gain":
                    genome_event_counts[genome_name]["gains"] += 1
                elif node.event == "loss":
                    genome_event_counts[genome_name]["losses"] += 1

    for node in t_w.traverse("postorder"):
        node_name = node.name if node.name else f"Node_{node.get_leaf_names()[0]}"
        if node_name not in node_event_counts:
            node_event_counts[node_name] = {"gains": 0, "losses": 0}
        if not node.is_root() and hasattr(node, "state") and hasattr(node.up, "state"):
            if node.state > node.up.state:
                node_event_counts[node_name]["gains"] += 1
            elif node.state < node.up.state:
                node_event_counts[node_name]["losses"] += 1

    for leaf in t_w.get_leaves():
        if not hasattr(leaf, "state") or not hasattr(leaf.up, "state"):
            continue
        genome = leaf.name
        if genome not in genome_parent_event_counts:
            genome_parent_event_counts[genome] = {"gains": 0, "losses": 0}
        if leaf.state > leaf.up.state:
            genome_parent_event_counts[genome]["gains"] += 1
        elif leaf.state < leaf.up.state:
            genome_parent_event_counts[genome]["losses"] += 1

    if i == 0:
        annotated_tree_wagner = t_w

print("End of the analysis")
print("Saving Wagner parsimony analysis results")
pd.DataFrame(results_wagner).to_csv("wagner_gain_loss_summary.csv", index=False)

print("Saving GoF/LoF events per genomes")
pd.DataFrame(genome_events).to_csv("gain_loss_per_genoma.csv", index=False)

print("Saving number of events per genoma...")
df_genome_summary = pd.DataFrame([
    {"taxon": g, "gains": v["gains"], "losses": v["losses"]}
    for g, v in genome_event_counts.items()
])
df_genome_summary.to_csv("genoma_gain_loss_summary.csv", index=False)

print("Saving events GoF/LoF compared with genereting node")
df_parent_events = pd.DataFrame([
    {"taxon": g, "gains_from_parent": v["gains"], "losses_from_parent": v["losses"]}
    for g, v in genome_parent_event_counts.items()
])
df_parent_events.to_csv("genereting_nodes_events.csv", index=False)

print("Saving internal nodes events ...")
df_node_events = pd.DataFrame([
    {"node": node, "gains": counts["gains"], "losses": counts["losses"]}
    for node, counts in node_event_counts.items()
])
df_node_events.to_csv("nodo_eventi_gain_loss.csv", index=False)

print("Saving presence heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(matrix.loc[[leaf.name for leaf in tree.get_leaves()]], cmap="Greys", cbar=False)
plt.title("Heatmap presence/absence")
plt.xlabel("COGs family")
plt.ylabel("Genomes")
plt.xticks([], [])
plt.yticks([], [])
plt.tight_layout()
plt.savefig("heatmap_COGs.png", dpi=300)
plt.close()

def layout(node):
    if node.is_leaf():
        name_face = AttrFace("name", fsize=10)
        faces.add_face_to_node(name_face, node, column=0, position="branch-right")
    if hasattr(node, "event") and node.event in {"gain", "loss"}:
        color = "green" if node.event == "gain" else "red"
        nstyle = NodeStyle()
        nstyle["fgcolor"] = color
        nstyle["size"] = 8
        node.set_style(nstyle)

print("Saving phylogenetic tree with wagner parsimony")
ts = TreeStyle()
ts.show_leaf_name = False
ts.layout_fn = layout
ts.title.add_face(faces.TextFace("Tree with gain events (green) / loss events (red)", fsize=12), column=0)
annotated_tree_wagner.render("Gain_loss_wagner_tree.pdf", tree_style=ts)

print("Script end!")
