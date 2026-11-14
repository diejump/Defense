"""原LoSplit UGBA入口片段；保留作者划分与finetune2，不自写损失。"""
import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from models.construct import model_construct
from src.author_support import start_clock, elapsed, logits_of, capture_evaluation
from help_funcs import prune_unrelated_edge, prune_unrelated_edge_isolated

def run(inputs, args, device):
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
    idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
    known_nodes = torch.cat([idx_train,idx_attach]).to(device)
    reference_begin = start_clock(device)


    known_nodes = torch.cat([idx_train,idx_attach]).to(device)
    # edge weight for clean edge_index, may use later #
    edge_weight = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)

    # data_loaded = torch.load(args.pre_train_param, map_location='cpu')
    # idx_poison_found = data_loaded['idx_poison_found']
    # idx_clean_found = data_loaded['idx_clean_found']
    # target_label = data_loaded['target_label']
    # poison_x = data_loaded['poison_x']
    # poison_edge_index = data_loaded['poison_edge_index']
    # poison_edge_weights = data_loaded['poison_edge_weights']
    # poison_labels = data_loaded['poison_labels']

    if(args.defense_mode == 'prune'):
        poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
    elif(args.defense_mode == 'reconstruct'):
        poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,data.x,data.edge_index,device, idx_attach, large_graph=True)
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)

    elif(args.defense_mode == 'isolate'):
        poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
        bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
    else:
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
    print("precent of left attach nodes: {:.3f}"\
        .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))


    test_model = model_construct(args,args.test_model,data,device).to(device) 

    if test_model == 'ABL':
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False, num_attach=len(idx_attach))
    else:
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
        # test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=200,verbose=False, finetune7=True, target_label=args.target_class, num_attach=len(idx_attach), num_epoch=args.epochs)
        # test_model.fit(poison_x, poison_edge_index, poison_edge_weights, data.y, bkd_tn_nodes, idx_val,train_iters=200,verbose=False, finetune5=True)
        # test_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
    test_model.eval()

    output = logits_of(test_model(poison_x, poison_edge_index, poison_edge_weights))
    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])

    train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
    print("target class rate on Vs: {:.4f}".format(train_attach_rate))
    clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

    print("accuracy on clean test nodes: {:.4f}".format(clean_acc))


    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
    # output = test_model(induct_x,induct_edge_index,induct_edge_weights)
    train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
    print("target class rate on Vs: {:.4f}".format(train_attach_rate))
    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)

    output = logits_of(test_model(induct_x,induct_edge_index,induct_edge_weights))
    train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()

    asr = train_attach_rate
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()

    print("****After UGBA Attack****")
    print("ASR: {:.6f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
    print("Clean Accuracy: {:.6f}".format(clean_acc)) 
    induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
    output = output.cpu()


    ##LoSplit Defense###

    reference_seconds = elapsed(reference_begin, device)
    reference_metrics = {"CA": float(clean_acc), "ASR": float(flip_asr)}
    model_build_begin = start_clock(device)
    LoSplit= model_construct(args,'LoSplit',data,device, split_lr=args.split_lr).to(device) 
    poison_found, clean_found, deltas, target_labels = LoSplit.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, known_nodes, idx_val,train_iters=args.epochs,verbose=False, split_epoch=args.split_epoch, num_attach=len(idx_attach))

    best_delta = max(deltas[1:])  
    best_t = [i for i, val in enumerate(deltas) if val == best_delta][0] 
    idx_poison_found = poison_found[best_t]
    idx_clean_found = clean_found[best_t]
    target_label = target_labels[best_t]

    model_build_seconds = elapsed(model_build_begin, device)

    data_to_save = {
        'idx_poison_found': idx_poison_found,
        'idx_clean_found': idx_clean_found,
        'target_label': target_label,
        'poison_x': poison_x,
        'poison_edge_index': poison_edge_index,
        'poison_edge_weights': poison_edge_weights,
        'poison_labels': poison_labels
    }

    torch.save(data_to_save, args.pre_train_param)



    model_build_begin = start_clock(device)
    test_model1 = model_construct(args, args.test_model, data, device).to(device)
    #Decoupling-Forgetting
    test_model1.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=200, verbose=False, finetune2=True, attach=idx_poison_found, clean=idx_clean_found,target_label=target_label, gamma=args.gamma)

    model_build_seconds += elapsed(model_build_begin, device)
    evaluation_frame = capture_evaluation(test_model1, device)
    prediction_begin = start_clock(device)

    # # RIGBD Robust Training
    # test_model1.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=200, verbose=False, finetune1=True)   

    # #Discarding Target Nodes/Only use Clean nodes                                                                       
    # test_model1.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=200, verbose=False, finetune3=True, clean=idx_clean_found)   

    # # SCRUB Unlearning              
    # test_model1.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=200, verbose=False, finetune4=True, attach=idx_poison_found, clean=idx_clean_found, teacher_model=test_model)

    # # Restroing Original Label
    # test_model1.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=200, verbose=False, finetune5=True, attach=idx_poison_found, clean=idx_clean_found, clean_labels=data.y)

    # # Feature Reinitialization
    # test_model1.fit(poison_x, train_edge_index, None, poison_labels, bkd_tn_nodes, idx_val, train_iters=200, verbose=False, finetune6=True, attach=idx_poison_found, attach_feature=data.x)

    induct_edge_index = torch.cat([poison_edge_index, mask_edge_index], dim=1)
    induct_edge_weights = torch.cat([
        poison_edge_weights,
        torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)
    ])
    clean_acc = test_model1.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)

    induct_x, induct_edge_index, induct_edge_weights = model.inject_trigger(
        idx_atk, poison_x, induct_edge_index, induct_edge_weights, device
    )
    induct_x, induct_edge_index, induct_edge_weights = (
        induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()
    )

    output = logits_of(test_model1(induct_x, induct_edge_index, induct_edge_weights))
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk] == args.target_class).detach().cpu().numpy().mean()

    print("****After LoSplit Defense****")
    print("ASR: {:.6f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
    print("Clean Accuracy: {:.6f}".format(clean_acc)) 

    poison_common = set(idx_poison_found.tolist()) & set(idx_attach.tolist())
    clean_common = set(idx_clean_found.tolist()) & set(idx_train.tolist())

    TP = len(poison_common)   
    FP = len(idx_poison_found) - TP 
    FN = len(idx_attach) - TP 
    TN = len(clean_common) 

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    fpr = FP / (FP + TN + 1e-8)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"FPR: {fpr:.4f}")

    prediction_seconds = elapsed(prediction_begin, device)
    return {
        "models": [test_model1],
        "source_metrics": {"CA": float(clean_acc), "ASR": float(flip_asr),
            "CA_scope": "原作者在恢复测试关联边、尚未注入测试触发的图上评估idx_clean_test。",
            "ASR_scope": "原作者对全部idx_atk整体注入后，仅以原标签非目标类节点计算ASR。"},
        "reference_source_metrics": reference_metrics,
        "selected_nodes": (idx_poison_found).detach().cpu().tolist(),
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
    test_model1 = test_model
    induct_edge_index = torch.cat([poison_edge_index, mask_edge_index], dim=1)
    induct_edge_weights = torch.cat([
        poison_edge_weights,
        torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)
    ])
    clean_acc = test_model1.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)
    induct_x, induct_edge_index, induct_edge_weights = model.inject_trigger(
        idx_atk, poison_x, induct_edge_index, induct_edge_weights, device
    )
    induct_x, induct_edge_index, induct_edge_weights = (
        induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()
    )
    output = logits_of(test_model1(induct_x, induct_edge_index, induct_edge_weights))
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk] == args.target_class).detach().cpu().numpy().mean()

    return {"CA": clean_acc, "ASR": flip_asr,
        "CA_scope": "原作者在恢复测试关联边、尚未注入测试触发的图上评估idx_clean_test。",
        "ASR_scope": "原作者对全部idx_atk整体注入后，仅以原标签非目标类节点计算ASR。"}
