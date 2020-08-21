CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --ckpt_dir=./ckpt_coco \
  --pretrained_dir=./pretrained/model.ckpt \
  --mode=train \
  --tfrecords_dir= ./path/to/tfrecords/ \
  --image_size=384 \
  --max_boxes=100 \
  --num_train=32147 \
  --num_validation=7550 \
  --num_preprocess_threads=4 \
  --batch_size=16 \
  --learning_rate=0.0004 \
