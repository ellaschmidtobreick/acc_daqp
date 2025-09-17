import numpy as np

# Naive model
def naive_model(n_vector,m_vector,all_labels):
    # Compute node metrics
    preds = [0 for i in range(len(all_labels))]
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(all_labels)):
        tp += ((preds[i] == 1) & (all_labels[i] == 1)).sum().item()
        fp += ((preds[i] == 1) & (all_labels[i] == 0)).sum().item()
        tn += ((preds[i] == 0) & (all_labels[i] == 0)).sum().item()
        fn += ((preds[i] == 0) & (all_labels[i] == 1)).sum().item()

    acc = (tp + tn) / (tp + tn +fp + fn) if (tp + tn +fp + fn) > 0 else 0  
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    # Initialize
    graph_pred = []
    num_wrongly_pred_nodes_per_graph = []
    perc_wrongly_pred_nodes_per_graph = []
    
    # Compute graph metric
    current = 0
    correctly_predicted_graphs = 0
    for n, m in zip(n_vector, m_vector):
        all_labels_graph = np.array(all_labels[current:current+(n+m)])
        preds_graph = np.array(preds[current:current+(n+m)])     
        graph_pred.append(preds_graph == all_labels_graph)
        
        correctly_predicted_graphs += int(np.all(preds_graph == all_labels_graph))

        wrong = np.abs((n+m) - np.sum(all_labels_graph == preds_graph))
        num_wrongly_pred_nodes_per_graph.append(wrong)
        perc_wrongly_pred_nodes_per_graph.append([wrong / (n + m)])

        current += (n+m)
        
    return acc, prec, rec, f1, perc_wrongly_pred_nodes_per_graph, correctly_predicted_graphs