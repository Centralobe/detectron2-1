# Commands for the DGX

## Build Docker
```bash
cd detectron2
docker build -t detectron2 --build-arg USER_ID=1879402193 .
```

## Run the docker
```bash
docker run -it --name detectron2 \
  -v /home/cmanss/Agri-Gaia/data/maize_dataset:/home/appuser/data \
  -v /home/cmanss/Agri-Gaia/logs:/home/appuser/logs \
  --gpus all \
  detectron2:latest

docker exec -ti detectron2 /bin/bash
```

## In the Docker
Before running everything
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # choose wisely
cd detectron2
```
For regular config files (yaml)
```bash
python3 tools/train_maize.py --model retinanet_50 --data /home/appuser/data --output /home/appuser/logs --batch_size 32
```

For lazyconfig files
```bash
python3 tools/lazyconfig_train_maize.py \
  --config-file /home/appuser/detectron2/configs/COCO-Detection/fcos_R_50_FPN_1x_dgx_maize.py
```