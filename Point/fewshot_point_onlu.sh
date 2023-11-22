CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_5w10s --way 5 --shot 10 --fold 5
CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_5w20s --way 5 --shot 20 --fold 5
CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_10w10s --way 10 --shot 10 --fold 5
CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_10w20s --way 10 --shot 20 --fold 5
