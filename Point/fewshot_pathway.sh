CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_5w10s_audio --way 5 --shot 10 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-audio-0715.pth
CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_5w10s_image --way 5 --shot 10 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-image-0715.pth
CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_5w10s_video --way 5 --shot 10 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-video-0718.pth

CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_5w20s_audio --way 5 --shot 20 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-audio-0715.pth
CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_5w20s_image --way 5 --shot 20 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-image-0715.pth
CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_5w20s_video --way 5 --shot 20 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-video-0718.pth

CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_10w10s_audio --way 10 --shot 10 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-audio-0715.pth
CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_10w10s_image --way 10 --shot 10 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-image-0715.pth
CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_10w10s_video --way 10 --shot 10 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-video-0718.pth

CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_10w20s_audio --way 10 --shot 20 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-audio-0715.pth
CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_10w20s_image --way 10 --shot 20 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-image-0715.pth
CUDA_VISIBLE_DEVICES=0 python main_pathway.py --config cfgs/fewshot_pathway.yaml --finetune_model --exp_name pathway_10w20s_video --way 10 --shot 20 --fold 5 --pretrained_path ../multimodal_pathway/vit-b16-video-0718.pth
 
