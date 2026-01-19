# run pocket/protein and ligand encoder model
for r in {1..6} 
do
    path2weight="/ext/nhe/with_sample_packing/L_hyp/LigUnity_pocket_ranking/pocket_ranking_FEP/checkpoint_avg_41-50_5.pt"
    path2result="./result/pocket_ranking/FEP/repeat_{r}"
    CUDA_VISIBLE_DEVICES=0 bash test.sh FEP pocket_ranking ${path2weight} ${path2result}
    
    path2weight="/ext/nhe/with_sample_packing/L_hyp/LigUnity_protein_ranking/protein_ranking_FEP/checkpoint_avg_41-50_5.pt"
    path2result="./result/protein_ranking/FEP/repeat_{r}"
    CUDA_VISIBLE_DEVICES=0 bash test.sh FEP protein_ranking ${path2weight} ${path2result}
done

# get final prediction of our model
python ensemble_result.py FEP