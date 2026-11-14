"""完整保留作者MAD入口的防御与五种子评估；由专用脚本生成。"""
import numpy as np
import torch
import torch.nn.functional as F
from models.construct import model_construct
from Detect_encoder import cluster_encoder_detect, prune_anomalous_edges_by_class
from anomaly_detect import prune_high_fluctuation_nodes, prune_edge, anomaly_embedding_edge, gnn_prune_low_entropy_nodes, prune_anomaly_nodes_edges
from src.author_support import start_clock, elapsed, logits_of, capture_evaluation

def get_idx_clean_and_trigger(x, ori_x):
    num_clean_nodes = ori_x.size(0)
    num_total_nodes = x.size(0)

    idx_clean = torch.arange(num_clean_nodes, device=x.device)

    idx_trigger = torch.arange(num_clean_nodes, num_total_nodes, device=x.device)

    return idx_clean, idx_trigger

def calculate_entropy(features):
    features_min = features.min(dim=1, keepdim=True)[0]
    features_max = features.max(dim=1, keepdim=True)[0]
    features_normalized = (features - features_min) / (features_max - features_min + 1e-10)
    row_sums = features_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    probabilities = features_normalized / row_sums

    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    return entropy

def calculate_std(features):
    std = torch.std(features, dim=1)
    return std

def calculate_entropy_predict(probabilities):
    return -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)

def run(inputs, args, device):
    requested_model = args.test_model
    if args.defense_mode != "MAD":
        raise ValueError("此适配器只运行作者完整MAD模式。")
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
    idx_clean, idx_trigger = get_idx_clean_and_trigger(poison_x, data.x)
    native_models = []
    evaluation_frames = []
    internal_seeds = []
    prediction_seconds = 0.0
    _reference_begin = start_clock(device)
    if(args.defense_mode == 'prune'):
        poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
    elif(args.defense_mode == 'isolate'):
        poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
        bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
    elif(args.defense_mode == 'reconstruct'):
        poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,data.x,data.edge_index,device, idx_attach, large_graph=True)
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
    elif (args.defense_mode == 'MAD_feature'):
        poison_edge_index, poison_edge_weights = prune_high_fluctuation_nodes(args, poison_edge_index,
                                                                              poison_edge_weights, poison_x, device,
                                                                              large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
    elif (args.defense_mode == 'MAD_embedding'):
        encoder_x, y_pred, gcn_encoder = cluster_encoder_detect(args, poison_x, poison_labels, poison_edge_index,
                                                                device, idx_attach, idx_trigger, current_time)
        poison_edge_index, poison_edge_weights = prune_edge(args, poison_edge_index, poison_edge_weights, encoder_x,
                                                            device, large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
    elif (args.defense_mode == 'MAD_confidence'):
        poison_edge_index, poison_edge_weights, gnn_model, prediction_entropy_threshold = gnn_prune_low_entropy_nodes(
            args, poison_edge_index, poison_edge_weights,
            poison_x, poison_labels, device, large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
    elif (args.defense_mode == 'MAD'):
        encoder_x, y_pred, gcn_encoder = cluster_encoder_detect(args, poison_x, poison_labels, poison_edge_index,
                                                                device, idx_attach, idx_trigger, current_time)
        poison_edge_index, poison_edge_weights, gnn_model, prediction_entropy_threshold = prune_anomaly_nodes_edges(
            args, poison_edge_index, poison_edge_weights, poison_x, poison_labels, encoder_x, y_pred, device,
            large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
    else:
        bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
    reference_seconds = elapsed(_reference_begin, device)
    model_build_seconds = reference_seconds
    print("precent of left attach nodes: {:.3f}" \
          .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist())) / len(idx_attach)))

    import torch.nn.functional as F

    import torch.nn.functional as F

    # models = ['GCN','GAT', 'GraphSage']
    models = [requested_model]
    total_overall_asr = 0
    total_overall_ca = 0
    for test_model in models:
        args.test_model = test_model
        rs = np.random.RandomState(args.seed)
        seeds = rs.randint(1000, size=5)
        internal_seeds = seeds.tolist()
        overall_asr = 0
        overall_ca = 0
        for seed in seeds:
            args.seed = seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            print(args)
            _final_build_begin = start_clock(device)
            test_model = model_construct(args,args.test_model,data,device).to(device)
            test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
            model_build_seconds += elapsed(_final_build_begin, device)
            evaluation_frames.append(capture_evaluation(test_model, device))
            _prediction_begin = start_clock(device)

            output = logits_of(test_model(poison_x,poison_edge_index,poison_edge_weights))

            # probabilities = F.softmax(output, dim=1)
            # predicted_labels = probabilities.argmax(dim=1)
            # encoder_x = test_model.get_h(poison_x, poison_edge_index).clone().detach()

            train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
            #%%
            induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
            induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
            clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
            # clean_idx = idx_clean_test[(data.y[idx_clean_test] == args.target_class).nonzero().flatten()]
            # clean_probabilities = F.softmax(output[clean_idx], dim=1)
            # clean_entropy = calculate_entropy_predict(clean_probabilities).detach().cpu().numpy()

            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
            if (args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                induct_edge_index, induct_edge_weights = prune_unrelated_edge(args, induct_edge_index, induct_edge_weights,
                                                                              induct_x, device)
            elif (args.defense_mode == 'MAD_feature'):
                induct_edge_index, induct_edge_weights = prune_high_fluctuation_nodes(args, induct_edge_index,
                                                                                      induct_edge_weights, induct_x, device)
            elif (args.defense_mode == 'MAD_embedding'):
                encoder_x_test = gcn_encoder.get_h(induct_x, induct_edge_index,
                                                   induct_edge_weights).clone().detach()
                y_pred_test = gcn_encoder.test(induct_x, induct_edge_index, induct_edge_weights)
                induct_edge_index, induct_edge_weights = prune_edge(args, induct_edge_index, induct_edge_weights,
                                                                    encoder_x_test, device)
            elif (args.defense_mode == 'MAD_confidence'):
                with torch.no_grad():
                    output_defense,x = gnn_model(induct_x, induct_edge_index, induct_edge_weights)
                    probabilities = F.softmax(output_defense, dim=1)
                    predicted_labels = probabilities.argmax(dim=1)

                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                mask = (entropy < prediction_entropy_threshold)  # & target_class_mask
                keep_edges_mask = ~(mask[induct_edge_index[0]] | mask[induct_edge_index[1]])
                # Filter the edge_index by the edges we want to keep
                induct_edge_index = induct_edge_index[:, keep_edges_mask]
                # Filter the edge weights similarly
                induct_edge_weights = induct_edge_weights[keep_edges_mask]
            elif (args.defense_mode == 'MAD'):
                with torch.no_grad():
                    output_defense,x = gnn_model(induct_x, induct_edge_index, induct_edge_weights)
                    probabilities = F.softmax(output_defense, dim=1)

                feature_std = calculate_std(induct_x)
                high_feature_std_mask = feature_std > args.fluctuation_thrd
                encoder_x_test = gcn_encoder.get_h(induct_x, induct_edge_index,
                                                   induct_edge_weights).clone().detach()
                y_pred_test = gcn_encoder.test(induct_x, induct_edge_index, induct_edge_weights)
                high_dist_edges_mask = anomaly_embedding_edge(args, induct_edge_index, induct_edge_weights, encoder_x_test,
                                                              device)

                prediction_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                # target_class_mask = (predicted_labels == most_common_class)
                low_prediction_entropy_mask = (prediction_entropy < prediction_entropy_threshold)  # & target_class_mask

                keep_edges_mask = ~(
                        low_prediction_entropy_mask[induct_edge_index[0]] | high_feature_std_mask[induct_edge_index[0]] |
                        low_prediction_entropy_mask[induct_edge_index[1]] | high_feature_std_mask[induct_edge_index[1]])
                keep_edges_mask = keep_edges_mask.to(device)
                high_dist_edges_mask = high_dist_edges_mask.to(device)
                final_edges_mask = keep_edges_mask | high_dist_edges_mask
                induct_edge_index = induct_edge_index[:, final_edges_mask]
                # Filter the edge weights similarly
                induct_edge_weights = induct_edge_weights[final_edges_mask]

            output = logits_of(test_model(induct_x,induct_edge_index,induct_edge_weights))
            train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
            asr = train_attach_rate
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
            ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
            induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
            output = output.cpu()
            if args.defense_mode=='none':
                overall_asr += asr
            else:
                overall_asr += flip_asr
            # overall_asr += asr
            overall_ca += clean_acc
            test_model = test_model.cpu()
            native_models.append(test_model)
            prediction_seconds += elapsed(_prediction_begin, device)
        overall_asr = overall_asr / len(seeds)
        overall_ca = overall_ca / len(seeds)
        print("Overall ASR: {:.4f} ({} model, Seed: {})".format(overall_asr, args.test_model, args.seed))
        print("Overall Clean Accuracy: {:.4f}".format(overall_ca))

        total_overall_asr += overall_asr
        total_overall_ca += overall_ca
        test_model.to(torch.device('cpu'))
        torch.cuda.empty_cache()

    total_overall_asr = total_overall_asr/len(models)
    total_overall_ca = total_overall_ca/len(models)
    print("Total Overall ASR: {:.4f} ".format(total_overall_asr))
    print("Total Clean Accuracy: {:.4f}".format(total_overall_ca))

    # 原循环完成后仅将保留对象移至CPU，便于原生检查点保存。
    for native_model in native_models:
        native_model.cpu()
    return {"models": native_models,
            "evaluation_frames": evaluation_frames,
            "source_metrics": {"CA": float(total_overall_ca), "ASR": float(total_overall_asr),
                               "CA_scope": "作者clean_acc：净化训练边加未额外净化heldout边；五个内部种子均值",
                               "ASR_scope": "作者flip_asr：全部idx_atk中原标签非目标类别；触发图联合掩码后；五种子均值"},
            "defended_edges": poison_edge_index, "defended_weights": poison_edge_weights,
            "author_training_nodes": bkd_tn_nodes.cpu().tolist(),
            "author_attached_nodes": idx_attach.cpu().tolist(),
            "auxiliary_models": {"encoder": gcn_encoder, "confidence": gnn_model},
            "entropy_threshold": prediction_entropy_threshold,
            "source_args": vars(args).copy(), "internal_seeds": internal_seeds,
            "model_build_seconds": float(model_build_seconds),
            "reference_seconds": float(reference_seconds), "prediction_seconds": float(prediction_seconds),
            "timing_scope": {"reference": "作者两个辅助模型完整训练与训练图联合处理，计入model_build_seconds",
                             "model_build": "完整辅助阶段及原生五次最终模型构造和fit的累计",
                             "prediction": "五种子原生训练诊断前向、正常测试、触发注入与联合处理、测试和循环内CPU传输"}}


def evaluate(inputs, args, device, test_model, auxiliary=None, entropy_threshold=None):
    if args.defense_mode != "MAD":
        raise ValueError("此回调只评估作者完整MAD模式。")
    data = inputs["data"]
    poison_x = inputs["poison_x"]
    poison_edge_index = inputs["poison_edge_index"]
    poison_edge_weights = inputs["poison_edge_weights"]
    idx_attach = inputs["idx_attach"]
    idx_clean_test = inputs["idx_clean_test"]
    idx_atk = inputs["idx_atk"]
    mask_edge_index = inputs["mask_edge_index"]
    model = inputs["trigger"]
    gcn_encoder = auxiliary["encoder"]
    gnn_model = auxiliary["confidence"]
    prediction_entropy_threshold = entropy_threshold
    output = logits_of(test_model(poison_x,poison_edge_index,poison_edge_weights))

    # probabilities = F.softmax(output, dim=1)
    # predicted_labels = probabilities.argmax(dim=1)
    # encoder_x = test_model.get_h(poison_x, poison_edge_index).clone().detach()

    train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
    #%%
    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
    clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
    # clean_idx = idx_clean_test[(data.y[idx_clean_test] == args.target_class).nonzero().flatten()]
    # clean_probabilities = F.softmax(output[clean_idx], dim=1)
    # clean_entropy = calculate_entropy_predict(clean_probabilities).detach().cpu().numpy()

    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
    if (args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
        induct_edge_index, induct_edge_weights = prune_unrelated_edge(args, induct_edge_index, induct_edge_weights,
                                                                      induct_x, device)
    elif (args.defense_mode == 'MAD_feature'):
        induct_edge_index, induct_edge_weights = prune_high_fluctuation_nodes(args, induct_edge_index,
                                                                              induct_edge_weights, induct_x, device)
    elif (args.defense_mode == 'MAD_embedding'):
        encoder_x_test = gcn_encoder.get_h(induct_x, induct_edge_index,
                                           induct_edge_weights).clone().detach()
        y_pred_test = gcn_encoder.test(induct_x, induct_edge_index, induct_edge_weights)
        induct_edge_index, induct_edge_weights = prune_edge(args, induct_edge_index, induct_edge_weights,
                                                            encoder_x_test, device)
    elif (args.defense_mode == 'MAD_confidence'):
        with torch.no_grad():
            output_defense,x = gnn_model(induct_x, induct_edge_index, induct_edge_weights)
            probabilities = F.softmax(output_defense, dim=1)
            predicted_labels = probabilities.argmax(dim=1)

        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
        mask = (entropy < prediction_entropy_threshold)  # & target_class_mask
        keep_edges_mask = ~(mask[induct_edge_index[0]] | mask[induct_edge_index[1]])
        # Filter the edge_index by the edges we want to keep
        induct_edge_index = induct_edge_index[:, keep_edges_mask]
        # Filter the edge weights similarly
        induct_edge_weights = induct_edge_weights[keep_edges_mask]
    elif (args.defense_mode == 'MAD'):
        with torch.no_grad():
            output_defense,x = gnn_model(induct_x, induct_edge_index, induct_edge_weights)
            probabilities = F.softmax(output_defense, dim=1)

        feature_std = calculate_std(induct_x)
        high_feature_std_mask = feature_std > args.fluctuation_thrd
        encoder_x_test = gcn_encoder.get_h(induct_x, induct_edge_index,
                                           induct_edge_weights).clone().detach()
        y_pred_test = gcn_encoder.test(induct_x, induct_edge_index, induct_edge_weights)
        high_dist_edges_mask = anomaly_embedding_edge(args, induct_edge_index, induct_edge_weights, encoder_x_test,
                                                      device)

        prediction_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
        # target_class_mask = (predicted_labels == most_common_class)
        low_prediction_entropy_mask = (prediction_entropy < prediction_entropy_threshold)  # & target_class_mask

        keep_edges_mask = ~(
                low_prediction_entropy_mask[induct_edge_index[0]] | high_feature_std_mask[induct_edge_index[0]] |
                low_prediction_entropy_mask[induct_edge_index[1]] | high_feature_std_mask[induct_edge_index[1]])
        keep_edges_mask = keep_edges_mask.to(device)
        high_dist_edges_mask = high_dist_edges_mask.to(device)
        final_edges_mask = keep_edges_mask | high_dist_edges_mask
        induct_edge_index = induct_edge_index[:, final_edges_mask]
        # Filter the edge weights similarly
        induct_edge_weights = induct_edge_weights[final_edges_mask]

    output = logits_of(test_model(induct_x,induct_edge_index,induct_edge_weights))
    train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
    asr = train_attach_rate
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
    output = output.cpu()
    # overall_asr += asr
    test_model = test_model.cpu()

    return {"CA": clean_acc, "ASR": flip_asr,
            "CA_scope": "作者clean_acc：已保存净化训练边加未额外净化heldout边；当前一个原生模型",
            "ASR_scope": "作者flip_asr：原标签非目标类别的idx_atk；原测试联合掩码；当前一个原生模型"}
