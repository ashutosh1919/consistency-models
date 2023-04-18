import os
import gradio as gr
from image_sampling import *

checkpoint_dir = 'checkpoints'

models = []
checkpoints = []

for root, dirs, files in os.walk(checkpoint_dir):
    for file in files:
        if file.endswith(".pt"):
            models.append(file.split('.')[0])
            checkpoints.append(os.path.abspath(os.path.join(root, file)))

def get_edm_default():
    return dict(
        training_mode='edm',
        generator='determ-indiv',
        batch_size=8,
        sigma_max=80,
        sigma_min=0.002,
        s_churn=0,
        steps=40,
        sampler='heun',
        model_path='',
        attention_resolutions="32,16,8",
        class_cond=False,
        dropout=0.1,
        image_size=256,
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        num_samples=8,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=False,
        weight_schedule='karras'
    )
            
def get_edm_lsun_defaults():
    return get_edm_default()
    

def get_edm_imagenet_defaults():
    defaults = get_edm_default()
    defaults.update(dict(
        batch_size=64,
        generator='determ',
        class_cond=True,
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        use_scale_shift_norm=True
    ))
    return defaults

def get_ct_imagenet_defaults():
    defaults = get_edm_default()
    defaults.update(dict(
        training_mode='consistency_distillation',
        sampler='onestep',
        batch_size=256,
        class_cond=True,
        use_scale_shift_norm=True,
        dropout=0.0,
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        weight_schedule='uniform'
    ))
    return defaults

def get_ct_lsun_defaults():
    defaults = get_edm_default()
    defaults.update(dict(
        training_mode='consistency_distillation',
        sampler='onestep',
        batch_size=32,
        dropout=0.0,
        weight_schedule='uniform'
    ))
    return defaults


def sample_consistency_images(
    model_name,
    generator,
    steps,
    dropout
):
    model_index = models.index(model_name)
    model_info = model_name.split('_')
    if model_info[0] == 'edm':
        if model_info[1] in ['bedroom256', 'cat256']:
            args = get_edm_lsun_defaults()
        elif model_info[1] == 'imagenet64':
            args = get_edm_imagenet_defaults()
    elif model_info[0] in ['ct', 'cd']:
        if model_info[1] in ['bedroom256', 'cat256']:
            args = get_ct_lsun_defaults()
        elif model_info[1] == 'imagenet64':
            args = get_ct_imagenet_defaults()
    args.update(dict(
        model_path=str(checkpoints[model_index]),
        generator=generator,
        steps=steps,
        dropout=dropout
    ))
    return sample_images(args)


app_inputs = [
    gr.components.Dropdown(models,
                           label="Model Name",
                           value='edm_bedroom256_ema'),
    gr.components.Dropdown(['determ', 'determ-indiv', 'dummy'],
                           label="Generator",
                           value='determ'),
    gr.Number(label="Steps",
              value=40,
              precision=0),
    gr.Slider(label="Dropout",
              minimum=0.,
              maximum=0.9,
              value=0.1)
]

gal = gr.Gallery(label='Generated Images')
gal.style(columns=2)

demo = gr.Interface(
    fn=sample_consistency_images,
    inputs=app_inputs,
    outputs=[gal],
)

demo.launch(share=True)
