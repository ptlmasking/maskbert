declare -a list_of_lr=(1e-4)
declare -a list_of_weight_decay=(0)
declare -a list_of_optimizers=('adam')
declare -a list_of_manual_seed=(1)

for ((i=0;i<${#list_of_lr[@]};++i)); do
    for ((j=0;j<${#list_of_weight_decay[@]};++j)); do
        for ((k=0;k<${#list_of_optimizers[@]};++k)); do
            for ((sd=0;sd<${#list_of_manual_seed[@]};++sd)); do
                python main.py --override False --max_seq_len 128 \
                    --experiment test \
                    --ptl distilbert \
                    --model distilbert-base-uncased \
                    --task swag \
                    --model_scheme multiplechoice \
                    --optimizer ${list_of_optimizers[k]} --lr_for_mask ${list_of_lr[i]} --lr 1e-5 --batch_size 32 --weight_decay ${list_of_weight_decay[j]} \
                    --do_BL True \
                    --do_MS False \
                    --ptl_req_grad True \
                    --classifier_req_grad True \
                    --mask_classifier False \
                    --mask_ptl False \
                    --world 0 \
                    --manual_seed ${list_of_manual_seed[sd]} \
                    --num_epochs 10 \
                    --eval_every_batch 60 --early_stop 2 \
                    --name_of_masker MaskedLinear1 \
                    --layers_to_mask 2,3,4,5,6,7,8,9,10,11 \
                    --masking_scheduler_conf lambdas_lr=0,sparsity_warmup=automated_gradual_sparsity,final_sparsity=0.03,sparsity_warmup_interval_epoch=0.5,init_epoch=0,final_epoch=0
            done
        done
    done
done
