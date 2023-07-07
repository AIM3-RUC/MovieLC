This Code refers to the work [Livebot](https://github.com/lancopku/livebot), [VideoIC](https://github.com/AIM3-RUC/VideoIC/tree/master) and [UniVL](https://github.com/microsoft/UniVL)
# Requirements
+ python>=3.7.0
+ torch==1.7.1
+ numpy==1.21.6
+ tqdm==4.65.0
+ boto3==1.27.0
+ requests==2.31.0

# Run Model
## Preparing
Download `processed data` and put it under `MovieLC/data`. \
Come to the folder `cd MovieLC/src/KLVCG`

## Pretrain
We use external knowledge from both knowledge graph and comments of other videos in the pre-training stage.

    CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 main_task.py --do_pretrain --data_path ../../data/processed_data --setup_path setup_knowledge_KG_AC --output_dir ../../data/output --runid klvcg_movielc_pretrain --batch_size 128 --batch_size_val 128
## Finetune
To avoid noise, we only use external knowledge from comments of other videos in the fine-tunening stage.

    CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 main_task.py --do_train --data_path --data_path ../../data/processed_data --setup_path setup_knowledge_AC --output_dir ../../data/output --runid klvcg_movielc_finetune --init_model ../../data/output/klvcg_movielc_pretrain/pytorch_model.bin.BEST --batch_size 128 --batch_size_val 128
## Test
Test the model.

    CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 main_task.py --do_test --data_path ../../data/processed_data --setup_path setup_knowledge_AC --output_dir ../../data/output --runid klvcg_movielc_checkpoint --init_model ../../data/output/klvcg_movielc_checkpoint/pytorch_model.bin.BEST
## Generate
Generate comments for visualization.

    CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 main_task.py --do_gen --data_path ../../data/processed_data --setup_path setup_knowledge_AC --output_dir ../../data/output --runid klvcg_movielc_checkpoint --init_model ../../data/output/klvcg_movielc_checkpoint/pytorch_model.bin.BEST
# Process Data
## Add candidates for test set
## Add knowledge for setup files