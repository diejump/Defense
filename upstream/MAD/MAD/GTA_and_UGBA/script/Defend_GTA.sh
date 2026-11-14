models=(GCN)
#defense_modes=(none entropy_feature entropy_predict entropy encoder cmp_encoder)
#defense_modes=(none MAD_feature MAD_embedding MAD_confidence MAD)
defense_modes=(MAD)

### # GTA: Cora
# for defense_mode in ${defense_modes[@]};
# do
#     for model in ${models[@]};
#     do
#         python -u run_GTA.py \
#             --dist_thrd=0.4\
#             --fluctuation_thrd=0.015\
#             --defense_thrd=3\
#             --trigger_size=3\
#             --vs_size=10\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=Cora\
#             --bkd_model=GTA
#     done
# done
##
#### GTA: Pubmed
# for defense_mode in ${defense_modes[@]};
# do
#     for model in ${models[@]};
#     do
#         python -u run_GTA.py \
#             --dist_thrd=0.1\
#             --fluctuation_thrd=0.013\
#             --vs_size=40\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=Pubmed\
#             --defense_thrd=4\
#             --bkd_model=GTA
#     done
# done
####
##### # GTA: Flickr
 for defense_mode in ${defense_modes[@]};
 do
     for model in ${models[@]};
     do
         python -u run_GTA.py \
             --dist_thrd=0.5\
             --fluctuation_thrd=0.02\
             --train_lr=0.02 \
             --hidden=32 \
             --vs_size=80\
             --test_model=${model}\
             --defense_mode=${defense_mode}\
             --epochs=200\
             --dataset=Flickr\
             --defense_thrd=0.3\
             --bkd_model=GTA
     done
 done
####
#### # GTA: ogbn
# for defense_mode in ${defense_modes[@]};
# do
#     for model in ${models[@]};
#     do
#         python -u run_GTA.py \
#             --train_lr=0.02 \
#             --dist_thrd=2\
#             --fluctuation_thrd=0.3\
#             --vs_size=160\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=ogbn-arxiv\
#             --defense_thrd=0.3\
#             --bkd_model=GTA
#     done
# done
#
