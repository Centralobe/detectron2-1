# Commands for the DGX

## Build Docker
```bash
cd detectron2
docker build -t detectron2:v0 --build-arg USER_ID=$UID .
```

## Run the docker
```bash
docker run -it --name detectron2 \
  -v /home/cmanss/Agri-Gaia/data:/home/appuser/data \
  -v /home/cmanss/Agri-Gaia/logs:/home/appuser/logs \
  --gpus all \
  detectron2:v0

docker exec -ti detectron2 /bin/bash
```

## In the Docker
Before running everything
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # choose wisely
```
For regular config files (yaml)
```bash
python3 tools/train_maize.py --model retinanet_50 --data /home/appuser/data --output /home/appuser/logs --batch_size 32
```

For lazyconfig files
```bash
python3 tools/lazyconfig_train_net.py \
  --config-file /home/appuser/detectron2_repo/configs/COCO-Detection/<CONFIG_FILE>
```
*Configfiles:*
* retinanet_R_50_FPN_1x_dgx_maize.py
* retinanet_R_50_FPN_3x_dgx_maize.py
* fcos_R_50_FPN_1x_dgx_maize.py