##models=(GCN GraphSage GAT)
models=(GCN)
## isolate means the Prune+LD defense method
defense_modes=(none)
## #Cora
for defense_mode in ${defense_modes[@]};
do
    for model in ${models[@]};
    do
        python -u run_clean.py \
            --prune_thr=3\
            --dataset=Cora\
            --homo_loss_weight=50\
            --vs_number=10\
            --test_model=${model}\
            --defense_mode=${defense_mode}\
            --selection_method=cluster_degree\
            --homo_boost_thrd=0.5\
            --epochs=200\
            --trojan_epochs=400\
            --defense_thrd=3\
            --feature_thrd=4\
            --bkd_model=clean
    done
done
###
#### # Pubmed
 for defense_mode in ${defense_modes[@]};
 do
     for model in ${models[@]};
     do
         python -u run_clean.py \
             --prune_thr=0.1\
             --dataset=Pubmed\
             --homo_loss_weight=100\
             --vs_number=40\
             --test_model=${model}\
             --defense_mode=${defense_mode}\
             --selection_method=cluster_degree\
             --homo_boost_thrd=0.5\
             --epochs=200\
             --trojan_epochs=2000\
             --defense_thrd=4\
             --feature_thrd=5\
             --bkd_model=clean
     done
 done
###
### # Flickr
 for defense_mode in ${defense_modes[@]};
 do
     for model in ${models[@]};
     do
         python -u run_clean.py \
             --prune_thr=0.5\
             --dataset=Flickr\
             --hidden=64 \
             --homo_loss_weight=100\
             --vs_number=80\
             --test_model=${model}\
             --defense_mode=${defense_mode}\
             --selection_method=cluster_degree\
             --homo_boost_thrd=0.8\
             --epochs=200\
             --trojan_epochs=400\
             --defense_thrd=0.3\
             --feature_thrd=6\
             --bkd_model=clean
     done
 done
#
## # OGBN-Arixv
 for defense_mode in ${defense_modes[@]};
 do
     for model in ${models[@]};
     do
         python -u run_clean.py \
             --prune_thr=2\
             --dataset=ogbn-arxiv\
             --homo_loss_weight=200\
             --vs_number=160\
             --test_model=${model}\
             --defense_mode=${defense_mode}\
             --selection_method=cluster_degree\
             --homo_boost_thrd=0.8\
             --epochs=800\
             --trojan_epochs=800\
             --defense_thrd=0.3\
             --feature_thrd=5\
             --bkd_model=clean
     done
 done
