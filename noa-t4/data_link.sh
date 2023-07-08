train=data/train-300
mkdir -p $train

echo "Linking image folder ..."
ln -s ~/Documents/code/static/noa-t4-multi/train/noa $train
ln -s ~/Documents/code/static/noa-t4-multi/train/t4 $train

echo "Linking image folder complete."
