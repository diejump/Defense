##models=(GCN GraphSage GAT)
models=(GCN)
## isolate means the Prune+LD defense method
#defense_modes=(MAD_feature MAD_embedding MAD_confidence MAD)
defense_modes=(MAD)
### #Cora 0.015
#for defense_mode in ${defense_modes[@]};
#do
#        python -u run_adaptive.py \
#              --dist_thrd=0.5\
#              --fluctuation_thrd=0.015\
#              --dataset=Cora\
#              --k=50\
#              --vs_number=15\
#              --hidden=32\
#              --train_lr=0.01\
#              --lr=0.01\
#              --test_model=GCN\
#              --defense_mode=${defense_mode}\
#              --weight_targetclass=3\
#              --weight_target=1\
#              --weight_ood=1\
#              --epochs=200\
#              --trigger_size=3\
#              --trojan_epochs=400\
#              --defense_thrd=3\
#              --bkd_model=DPGBA
#done
#####
###### # Pubmed
# for defense_mode in ${defense_modes[@]};
# do
#         python -u run_adaptive.py \
#                --dist_thrd=0.1\
#                --fluctuation_thrd=0.013\
#                --dataset=Pubmed\
#                --hidden=32\
#                --vs_number=40\
#                --test_model=GCN\
#                --defense_mode=${defense_mode}\
#                --train_lr=0.01\
#                --lr=0.01\
#                --weight_ood=1\
#                --weight_targetclass=20\
#                --weight_target=1\
#                --trigger_size=3\
#                --range=0.1\
#                --epochs=200\
#                --trojan_epochs=400\
#                --defense_thrd=0.2\
#                --bkd_model=DPGBA
# done
##########
########## # Flickr defense_thrd=5
 for defense_mode in ${defense_modes[@]};
 do
         python -u run_adaptive.py \
            --dist_thrd=1.75\
            --fluctuation_thrd=0.04\
            --dataset=Flickr\
            --hidden=64\
            --vs_number=160\
            --test_model=GCN\
            --defense_mode=${defense_mode}\
            --train_lr=0.01\
            --lr=0.01\
            --weight_targetclass=20\
            --weight_ood=1\
            --weight_target=3\
            --trigger_size=3\
            --range=0.01\
            --epochs=200\
            --trojan_epochs=400\
            --defense_thrd=0.8\
            --bkd_model=DPGBA
 done
 ########## # Flickr defense_thrd=5
