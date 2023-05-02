python /users/xpb20111/programs/benchmark_models/ml_script_refacNEW2_ObjOri3.py \
       -m   Ridge \
       -rfe False \
       -f /users/xpb20111/Prosperity_partnership/GSK_data/HBRD4_BD1_LIG_FRET_FC/HBRD4_BD1_LIG_FRET_FC_4_RDKit2D_RDKitTauto_desc.pk \
       -col x y c \
       -o Ridge_scores.dat \
       -mc_cv 2 \
       -hyper_cv 2 \
       -refit
