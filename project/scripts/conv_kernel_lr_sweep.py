import itertools, subprocess, os

# Focused ConvVAE kernel+LR sweep: use best latent_dim & hidden_channels from previous sweep
kernels = [3, 5]
lrs = ['1e-3', '5e-4']
ld = 64
hc = 32
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for k, lr in itertools.product(kernels, lrs):
    out = f'results/conv_kernel_lr_sweep/conv_k{k}_lr{lr.replace(".","p")}'
    os.makedirs(out, exist_ok=True)
    print('Training ConvVAE kernel', k, 'lr', lr)
    train_script = os.path.join(basedir, 'src', 'train.py')
    subprocess.check_call([
        'python', train_script,
        '--feat_dir', os.path.join(basedir, 'data', 'features'),
        '--out_dir', out,
        '--epochs', '40',
        '--batch_size', '4',
        '--model', 'conv',
        '--latent_dim', str(ld),
        '--hidden_channels', str(hc),
        '--lr', lr,
        '--kernel_size', str(k)
    ])
    print('Clustering', out)
    cluster_script = os.path.join(basedir, 'scripts', 'cluster_and_visualize.py')
    subprocess.check_call(['python', cluster_script, '--latents', f'{out}/latents.npy', '--k', '2', '--out_dir', out + '_analysis', '--metadata', os.path.join(basedir, 'data', 'raw', 'metadata_aligned.csv'), '--feat_dir', os.path.join(basedir, 'data', 'features', 'spec')])
print('Conv kernel+LR sweep done')
