#models=(GCN GraphSage GAT)
models=(GCN)
## isolate means the Prune+LD defense method
#defense_modes=(none MAD_feature MAD_embedding MAD_confidence MAD)
defense_modes=(MAD)
## #Cora
#for defense_mode in ${defense_modes[@]};
#do
#    for model in ${models[@]};
#    do
#        python -u run_adaptive.py \
#            --dist_thrd=3\
#            --fluctuation_thrd=0.015\
#            --dataset=Cora\
#            --homo_loss_weight=50\
#            --vs_number=10\
#            --test_model=${model}\
#            --defense_mode=${defense_mode}\
#            --selection_method=cluster_degree\
#            --homo_boost_thrd=0.5\
#            --epochs=200\
#            --trojan_epochs=400\
#            --defense_thrd=3\
#            --bkd_model=UGBA
#    done
#done
####
#### # Pubmed
# for defense_mode in ${defense_modes[@]};
# do
#     for model in ${models[@]};
#     do
#         python -u run_adaptive.py \
#             --dist_thrd=0.1\
#             --fluctuation_thrd=0.013\
#             --dataset=Pubmed\
#             --homo_loss_weight=100\
#             --vs_number=40\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --selection_method=cluster_degree\
#             --homo_boost_thrd=0.5\
#             --epochs=200\
#             --trojan_epochs=2000\
#             --defense_thrd=4\
#             --bkd_model=UGBA
#     done
# done
##########
########## # Flickr
 for defense_mode in ${defense_modes[@]};
 do
     for model in ${models[@]};
     do
         python -u run_adaptive.py \
             --dist_thrd=0.5\
             --fluctuation_thrd=0.02\
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
             --defense_thrd=3\
             --bkd_model=UGBA
     done
 done
#######
####### # OGBN-Arixv
# for defense_mode in ${defense_modes[@]};
# do
#     for model in ${models[@]};
#     do
#         python -u run_adaptive.py \
#             --dist_thrd=2\
#             --fluctuation_thrd=0.3\
#             --dataset=ogbn-arxiv\
#             --homo_loss_weight=200\
#             --vs_number=160\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --selection_method=cluster_degree\
#             --homo_boost_thrd=0.8\
#             --epochs=800\
#             --trojan_epochs=800\
#             --defense_thrd=0.3\
#             --bkd_model=UGBA
#     done
# done


