#!/usr/bin/env bash
#rm logs/* -rf
#rm trained_model/* -rf



python src/facenet_train.py \
           --batch_size 15 \
           --gpu_memory_fraction 0.25 \
           --models_base_dir trained_model_2017_05_15_10_24 \
           --pretrained_model trained_model_2017_05_15_10_24/20170515-121856/model-20170515-121856.ckpt-182784 \
           --model_def models.nn2 \
           --logs_base_dir logs_test \
           --data_dir /data/user_set/training/2017_05_15_10_24 \
           --lfw_pairs /data/user_set/lfw_pairs.txt \
           --image_size 224 \
           --lfw_dir /data/user_set/lfw \
           --optimizer ADAM \
           --evaluate_custom_pairs /data/user_set/person_000138_pairs.txt \
           --evaluate_custom_dir /data/user_set/person_000138 \
           --max_nrof_epochs 1000 \
           --learning_rate 0.00001
#           --data_dir /data/user_set/training/align_photo_png \
