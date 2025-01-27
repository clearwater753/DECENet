_base_ = [
    '_base_/models/decenet_decenet-d8.py',
    '_base_/datasets/bdd100k.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_80k.py'
]
custom_imports = dict(
    imports=['projects.bdd100k_dataset.mmseg.datasets.bdd100k'])
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)