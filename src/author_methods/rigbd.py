"""原RIGBD入口调用；仅参数化K/drop，保持原采样、目标及恢复流程。"""
import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from models.construct import model_construct
from src.author_support import start_clock, elapsed, logits_of, capture_evaluation

drop_ratio = 0.5

def sample_noise_all(edge_index, edge_weight, device):
    # Ensure inputs are on the correct device
    edge_index = edge_index.to(device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device)
    else:
        edge_weight = edge_weight.to(device)

    # Generate mask for edge dropping
    drop_mask = Bernoulli(1 - drop_ratio).sample(edge_weight.size()).bool()

    # Apply mask to edges
    noisy_edge_index = edge_index[:, drop_mask]
    noisy_edge_weight = edge_weight[drop_mask]

    # Get node degrees
    node_degrees = torch.zeros(edge_index.max() + 1, device=device)
    node_degrees.index_add_(0, noisy_edge_index[0], torch.ones(noisy_edge_index.size(1), device=device))
    # print('degree', node_degrees)

    # Restore edges for isolated nodes
    isolated_nodes = node_degrees == 0
    # print('isolated_nodes', isolated_nodes)
    if isolated_nodes.any():
        potential_restore_edges = isolated_nodes[edge_index[0]]
        # print('potential_restore_edges', potential_restore_edges)
        restore_edges = edge_index[:, potential_restore_edges]
        noisy_edge_index = torch.cat([noisy_edge_index, restore_edges], dim=1)
        restored_weights = torch.ones(restore_edges.size(1), device=device)
        noisy_edge_weight = torch.cat([noisy_edge_weight, restored_weights], dim=0)

    return noisy_edge_index, noisy_edge_weight


def find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, target_class):
    # Get the specific list to iterate through
    labels_list = poison_labels[bkd_tn_nodes[index_of_less_robust]]

    # Iterate through the list with index
    for i in range(len(labels_list) - 1):  # -1 to avoid index out of range
        if labels_list[i] != target_class and labels_list[i + 1] != target_class:
            return i - 1

    # Return None if the condition is not met in the loop
    return None


def run(inputs, args, device):
    global drop_ratio
    data = inputs["data"]
    poison_x = inputs["poison_x"]
    poison_edge_index = inputs["poison_edge_index"]
    poison_edge_weights = inputs["poison_edge_weights"]
    poison_labels = inputs["poison_labels"]
    idx_train = inputs["idx_train"]
    idx_attach = inputs["idx_attach"]
    idx_val = inputs["idx_val"]
    idx_clean_test = inputs["idx_clean_test"]
    idx_atk = inputs["idx_atk"]
    mask_edge_index = inputs["mask_edge_index"]
    model = inputs["trigger"]
    current_time = inputs["current_time"]
    reference_begin = start_clock(device)
    mask = data.y[idx_attach] != args.target_class
    mask = mask.to(device)
    print('number of poisoned target nodes', mask.sum())
    ## only attack those has groud truth labels != target_class ##
    idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
    known_nodes = torch.cat([idx_train,idx_attach]).to(device)
    predictions = []
    # edge weight for clean edge_index, may use later #
    edge_weight = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)



    #### train a backdoored model on poisoned graph #### 
    test_model = model_construct(args,args.test_model,data,device).to(device) 
    test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs, verbose=False)
    test_model.eval()
    clean_acc = test_model.test(poison_x,poison_edge_index, poison_edge_weights,poison_labels,idx_attach)
    output_clean = logits_of(test_model(poison_x,poison_edge_index,poison_edge_weights))
    ori_predict = torch.exp(output_clean[known_nodes])
    # print("accuracy on poisoned target nodes: {:.4f}".format(clean_acc))

    #### test backdoored model for comparison ####
    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()

    output = logits_of(test_model(induct_x,induct_edge_index,induct_edge_weights))
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    print("****Before Defense****")
    print("ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
    ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
    print("CA: {:.4f}".format(ca))



    reference_seconds = elapsed(reference_begin, device)
    reference_metrics = {"CA": float(ca), "ASR": float(flip_asr)}
    model_build_begin = start_clock(device)
    ###### formal test ########

    test_model = model_construct(args,args.test_model,data,device,add_self_loops=False).to(device) 
    test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
    test_model.eval()
    clean_acc = test_model.test(poison_x,poison_edge_index, poison_edge_weights,poison_labels,idx_attach)
    output_clean = logits_of(test_model(poison_x,poison_edge_index,poison_edge_weights))
    ori_predict = torch.exp(output_clean[known_nodes])
    print("accuracy on poisoned target nodes: {:.4f}".format(clean_acc))

    drop_ratio = args.drop_ratio

    predictions = []
    K=args.K
    for i in range(K):
                test_model.eval()
                noisy_poison_edge_index, noisy_poison_edge_weights = sample_noise_all(poison_edge_index, poison_edge_weights, device)
                output = logits_of(test_model(poison_x,noisy_poison_edge_index,noisy_poison_edge_weights))
                train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
                train_clean_rate = (output.argmax(dim=1)[idx_train]==data.y[idx_train]).float().mean()
                predictions.append(torch.exp(output[known_nodes]))

    epsilon = 1e-8
    deviations = []
    for sub_pred in predictions:
        sub_pred += epsilon
        deviation = F.kl_div(sub_pred.log(), ori_predict, reduce=False)
        deviations.append(deviation)

    summed_deviations = torch.zeros_like(deviations[0]).to(deviations[0].device)
    for deviation in deviations:
        ##### summed deviations for each node #####
        summed_deviations += deviation


    ##### get the index for nodes with less robustness #####
    
    ##### args.vs_number is unknown #####
    # index_of_less_robust = torch.sort(torch.mean(summed_deviations,dim=-1),descending=True)[1][:mask.sum()]
    index_of_less_robust = torch.sort(torch.mean(summed_deviations,dim=-1),descending=True)[1]


    # Example usage:
    # Assuming poison_labels, bkd_tn_nodes, index_of_less_robust, and target_class are defined
    result_index = find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, args.target_class)
    print("Index found:", result_index)
    # print(poison_labels[bkd_tn_nodes[index_of_less_robust][:result_index]])

    indexs = poison_labels[bkd_tn_nodes[index_of_less_robust][:result_index-1]]
    count = 0
    for i in indexs:
        if i == args.target_class:
            count += 1
    ## correct
    correct = count

    ## fasle
    false = len(indexs) - count



    test_model = model_construct(args,args.test_model,data,device).to(device) 

    # test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False, finetune=True, attach=bkd_tn_nodes[index_of_less_robust][:result_index])
    test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=400,verbose=False, finetune=True, attach=bkd_tn_nodes[index_of_less_robust][:result_index])

    model_build_seconds = elapsed(model_build_begin, device)
    evaluation_frame = capture_evaluation(test_model, device)
    prediction_begin = start_clock(device)
    ## test model on purified graph (poisoned target node) ##
    # clean_acc = test_model.test(poison_x,poison_edge_index, poison_edge_weights,poison_labels,idx_attach)

    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    # induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_attach,poison_x,induct_edge_index,induct_edge_weights,device)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()

    output = logits_of(test_model(induct_x,induct_edge_index,induct_edge_weights))
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    print("****After Defense****")
    print('recall', correct/mask.sum())
    print('Accuracy', correct/len(indexs))
    print("ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
    ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
    print("CA: {:.4f}".format(ca))




    prediction_seconds = elapsed(prediction_begin, device)
    return {
        "models": [test_model],
        "source_metrics": {"CA": float(ca), "ASR": float(flip_asr),
            "CA_scope": "原作者在已注入测试触发的整图上评估idx_clean_test。",
            "ASR_scope": "原作者对全部idx_atk整体注入后，仅以原标签非目标类节点计算ASR。"},
        "reference_source_metrics": reference_metrics,
        "selected_nodes": (bkd_tn_nodes[index_of_less_robust][:result_index]).detach().cpu().tolist(),
        "author_training_nodes": bkd_tn_nodes.cpu().tolist(),
        "author_attached_nodes": idx_attach.cpu().tolist(),
        "evaluation_frames": [evaluation_frame],
        "defended_edges": poison_edge_index,
        "defended_weights": poison_edge_weights,
        "auxiliary_models": {},
        "source_args": vars(args).copy(),
        "internal_seeds": [],
        "model_build_seconds": model_build_seconds,
        "reference_seconds": reference_seconds,
        "prediction_seconds": prediction_seconds,
    }


def evaluate(inputs, args, device, test_model, auxiliary=None, entropy_threshold=None):
    data = inputs["data"]
    poison_x = inputs["poison_x"]
    poison_edge_index = inputs["poison_edge_index"]
    poison_edge_weights = inputs["poison_edge_weights"]
    poison_labels = inputs["poison_labels"]
    idx_train = inputs["idx_train"]
    idx_attach = inputs["idx_attach"]
    idx_val = inputs["idx_val"]
    idx_clean_test = inputs["idx_clean_test"]
    idx_atk = inputs["idx_atk"]
    mask_edge_index = inputs["mask_edge_index"]
    model = inputs["trigger"]
    current_time = inputs["current_time"]
    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
    output = logits_of(test_model(induct_x,induct_edge_index,induct_edge_weights))
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    print("****After Defense****")
    print("ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
    ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

    return {"CA": ca, "ASR": flip_asr,
        "CA_scope": "原作者在已注入测试触发的整图上评估idx_clean_test。",
        "ASR_scope": "原作者对全部idx_atk整体注入后，仅以原标签非目标类节点计算ASR。"}
