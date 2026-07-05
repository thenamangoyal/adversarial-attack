INPUT_PATH="dataset/images"
MODIFIED_PATH="dataset/modified_images/""${1}"
CKPT_PATH='ckpts/'

python Test.py \
  --input_path="${INPUT_PATH}" \
  --modified_path="${MODIFIED_PATH}" \
  --checkpoint_path="${CKPT_PATH}"\
  --model="resnet_v2_152" \
  --image_width=224 \
  --image_height=224
