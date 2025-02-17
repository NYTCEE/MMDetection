import mmcv
from mmdet.apis import init_detector, inference_detector

config_file = './configs/scnet/scnet_r50_fpn_1x_coco.py'
checkpoint_file = './checkpoints/scnet_r50_fpn_1x_coco-c3f09857.pth'

device = 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

# Read image
img = mmcv.imread('demo/harrypotter.jpg')

# Get inference results
result = inference_detector(model, img)

# Draw and save results without visualization class
from mmdet.registry import VISUALIZERS
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# Draw the results
visualizer.add_datasample(
    name='result',
    image=img,
    data_sample=result,
    draw_gt=False,
    wait_time=0
)

# Save the visualization result
img_vis = visualizer.get_image()
mmcv.imwrite(img_vis, 'demo/seg_harrypotter_result.jpg')