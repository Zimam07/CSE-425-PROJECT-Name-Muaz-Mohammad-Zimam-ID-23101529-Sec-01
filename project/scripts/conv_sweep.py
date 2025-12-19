import itertools, subprocess, os

comb=list(itertools.product([8,16,32],[16,32]))
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for ld, hc in comb:
    out=f'results/conv_sweep/conv_ld{ld}_hc{hc}'
    os.makedirs(out, exist_ok=True)
    print('Training ConvVAE latent',ld,'hc',hc)
    train_script = os.path.join(basedir, 'src', 'train.py')
    subprocess.check_call(['python', train_script, '--feat_dir', os.path.join(basedir, 'data', 'features'), '--out_dir', out, '--epochs', '30', '--batch_size', '2', '--model', 'conv', '--latent_dim', str(ld), '--hidden_channels', str(hc), '--lr', '1e-3'])
    print('Clustering',out)
    cluster_script = os.path.join(basedir, 'scripts', 'cluster_and_visualize.py')
    subprocess.check_call(['python', cluster_script, '--latents', f'{out}/latents.npy', '--k', '2', '--out_dir', out+'_analysis'])
print('Conv sweep done')
