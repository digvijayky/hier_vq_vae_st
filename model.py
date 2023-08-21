import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
import math
import sys
import glob
import wandb
import umap
import pickle
import datetime
import sklearn.cluster as cluster
import pandas as pd
import pytorch_msssim

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from pdb import set_trace
from torch.optim import Adam
from torchvision.transforms import functional as tvf

rand_seed = 9
torch.manual_seed(rand_seed)


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay=0.0, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon


    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (
            loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encoding_indices,
        )


    def quantize_encoding_indices(self, encoding_indices, target_shape, device):
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=device
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(target_shape)
        return quantized.permute(0, 3, 1, 2).contiguous()


def positional_encoding(x, num_frequencies=6, incl_input=True):
    results = []
    if incl_input: results.append(x)
    if num_frequencies == 0: return x
    pos_res = []
    for L in range(num_frequencies):
        l_sq_pi = 2. ** L * np.pi
        pos_res.append(torch.sin(l_sq_pi * x))
        pos_res.append(torch.cos(l_sq_pi * x))
    pos_res = torch.cat(pos_res, dim=-1)
    results.append(pos_res)
    return torch.cat(results, dim=-1)


class hierast(nn.Module):
    def __init__(
        self,
        image_channels=3,
        num_layers=4,
        kernel_size=2,
        stride=2,
        padding=0,
        img_size=96,
        small_conv=False,
        embedding_dim=64,
        max_filters=512,
        use_max_filters=False,
        num_embeddings=512,
        commitment_cost=0.25,
        decay=0.99,
        scales=[0, 1, 3]
    ):
        super(hierast, self).__init__()

        self.scales = scales
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for scale_ in self.scales:
            if small_conv:
                scale_ += 1
            if not use_max_filters:
                max_filters = embedding_dim
            channel_sizes = self.calculate_channel_sizes(
                image_channels, max_filters, scale_
            )

            encoder_layers = nn.ModuleList()
            for i, (in_channels, out_channels) in enumerate(channel_sizes):
                if small_conv and i == 0:
                    encoder_layers.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        )
                    )
                else:
                    encoder_layers.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False,
                        )
                    )
                encoder_layers.append(nn.BatchNorm2d(out_channels))
                encoder_layers.append(nn.ReLU())

            if use_max_filters:
                encoder_layers.append(
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=embedding_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            self.encoders.append(nn.Sequential(*encoder_layers))

            self.vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )

            decoder_layers = nn.ModuleList()
            if use_max_filters:
                decoder_layers.append(
                    nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            for i, (out_channels, in_channels) in enumerate(channel_sizes[::-1]):
                if small_conv and i == scale_ - 1:
                    decoder_layers.append(
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        )
                    )
                else:
                    decoder_layers.append(
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False,
                        )
                    )
                decoder_layers.append(nn.BatchNorm2d(out_channels))
                if i != scale_ - 1:
                    decoder_layers.append(nn.ReLU())
                else:
                    decoder_layers.append(nn.Sigmoid())
            self.decoders.append(nn.Sequential(*decoder_layers))


    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes


    def forward(self, x):
        total_loss = 0.
        total_perplexity = 0.
        encoding_indices_list = []
        reconstructed = torch.zeros_like(x)
        for encoder, decoder in zip(self.encoders, self.decoders):
            encoded = encoder(x)
            loss, quantized, perplexity, encodings = self.vq_vae(encoded)
            total_loss += loss
            total_perplexity += perplexity
            encoding_indices_list.append(encodings)

            reconstructed += decoder(quantized)
        reconstructed /= len(self.scales)

        return loss, reconstructed, perplexity, encoding_indices_list


    def quantize_and_decode(self, x, target_shape, device):
        quantized = self.vq_vae.quantize_encoding_indices(x, target_shape, device)
        return self.decoder(quantized)


def mse_loss(reconstructed_x, x, use_sum=False):
    if use_sum:
        return nn.functional.mse_loss(reconstructed_x, x, reduction="sum")
    else:
        return nn.functional.mse_loss(reconstructed_x, x, reduction="mean")


def mse_ssim_loss(reconstructed_x, x, use_sum=False, ssim_module=None, mse_weight=1, ssim_weight=1):
    mse = mse_weight * mse_loss(reconstructed_x, x, use_sum)
    if ssim_module:
        ssim = ssim_weight * (1 - ssim_module(reconstructed_x, x))
    else:
        ssim = torch.tensor(0)
    return mse + ssim, {"MSE": mse.item(), "SSIM": ssim.item()}


def train_hier_vq_vae(dataset='vizgen_oncology',
                      dataset_dir='datasetsg/vizgen_data/oncology_data/',
                      image_channels=527,
                      nval_splits=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sample_list = np.array(sorted(glob.glob(f'{dataset_dir}/Human*')))
    img_size = 256
    batch_size = 4
    embedding_dim = 64
    num_embeddings = 8192
    num_layers = 3
    encoding_dim = img_size // (2 ** num_layers)
    crossval_idx = 1
    development = False
    scales = [0, 1, 3]
    notes = 'hierarchical vq vae'

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    run_name = f'hvqvae_{timestamp}'
    checkpoint_dir = os.path.join('checkpoints', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    figdir = os.path.join(checkpoint_dir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    epochs = 300

    model = hierast(image_channels=image_channels,
                    num_layers=num_layers,
                    img_size=img_size,
                    small_conv=True,
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    commitment_cost=.25,
                    use_max_filters=True,
                    max_filters=image_channels,
                    scales=scales)

    model = model.to(device)
    num_model_params = sum(p.numel() for p in model.parameters())

    kfold_cv = KFold(n_splits=nval_splits, random_state=None)
    train_subset, val_subset = list(kfold_cv.split(sample_list))[crossval_idx]
    train_subset, val_subset = sample_list[train_subset], sample_list[val_subset]
    pd.DataFrame(train_subset).to_csv(os.path.join(checkpoint_dir, 'train_subset.csv'), index=False, header=False)
    pd.DataFrame(val_subset).to_csv(os.path.join(checkpoint_dir, 'val_subset.csv'), index=False, header=False)

    run = wandb.init(
        project = 'hvqvae_vizgen_oncology',
        job_type = 'train_model',
        name = run_name,
        config = dict(dataset_dir=dataset_dir,
                      img_size=img_size,
                      image_channels=image_channels,
                      batch_size=batch_size,
                      embedding_dim=embedding_dim,
                      num_embeddings=num_embeddings,
                      num_layers=num_layers,
                      encoding_dim=encoding_dim,
                      num_model_params=num_model_params,
                      rand_seed=rand_seed,
                      num_training_samples=len(train_subset),
                      development=development,
                      notes=notes,
                      dataset=dataset))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=.98)
    ssim_module = pytorch_msssim.SSIM(data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03), channel=image_channels)

    xy_coords = torch.stack(torch.meshgrid(torch.arange(img_size), torch.arange(img_size)), dim=2).reshape(-1, 2) / 255.
    posenc_coords = positional_encoding(xy_coords, num_frequencies=6, incl_input=True)
    posenc_coords = posenc_coords.reshape((img_size, img_size, -1)).permute(2,0,1)

    for epoch in range(epochs):
        model.train()
        codebook_mappings_all_data = torch.zeros(num_embeddings, dtype=torch.int64, device=device)
        codebook_mappings_per_sample = torch.zeros(num_embeddings, dtype=torch.int64, device=device)
        codebook_mappings_per_sample_dict = {}
        for sample in train_subset:
            patient_name = os.path.basename(sample)
            cell_centroids = torch.load(os.path.join(sample, 'all_detected_cells_tiled_256_res_tensor.pt')).unsqueeze(1).to(torch.uint8)
            if cell_centroids.is_sparse: cell_centroids = cell_centroids.to_dense()

            posenc_cell_centroids = posenc_coords.repeat(cell_centroids.shape[0], 1, 1, 1)
            posenc_cell_centroids[(cell_centroids == 0).repeat(1, posenc_cell_centroids.shape[1], 1, 1)] = 0.

            det_transcripts = torch.load(os.path.join(sample, 'all_detected_transcripts_tiled_256_res_tensor.pt'))
            if det_transcripts.is_sparse: det_transcripts = det_transcripts.to_dense()
            images = torch.cat((cell_centroids, det_transcripts), dim=1)
            codebook_mappings_all_data += codebook_mappings_per_sample
            codebook_mappings_per_sample_dict[patient_name] = codebook_mappings_per_sample
            codebook_mappings_per_sample = torch.zeros(num_embeddings, dtype=torch.int64, device=device)

            images = images[torch.randperm(images.shape[0])]
            training_images = torch.split(images, split_size_or_sections=batch_size, dim=0)
            training_pe = torch.split(posenc_cell_centroids, split_size_or_sections=batch_size, dim=0)
            for idx, batch in enumerate(training_images):
                optimizer.zero_grad()
                batch = torch.cat((training_pe[idx], batch), dim=1).float().to(device)
                vq_loss, reconstructed, perplexity, encodings = model(batch)
                batch_loss, loss_dict = mse_ssim_loss(reconstructed, batch, use_sum=False, ssim_module=ssim_module, mse_weight=1, ssim_weight=1,)
                batch_loss += vq_loss
                batch_loss.backward()
                optimizer.step()
                codebook_mappings_in_batch = torch.bincount(torch.cat(encodings).flatten(), minlength=num_embeddings)
                codebook_mappings_per_sample += codebook_mappings_in_batch
                wandb.log({'batch_loss': batch_loss, 'vq_loss': vq_loss, 'perplexity': perplexity})
        sched.step()
        torch.save(model.state_dict(), f"{checkpoint_dir}/checkpoint_{epoch}.pt")

        with open(os.path.join(checkpoint_dir, f'ce_sample_dict_{epoch}.pkl'), 'wb') as f: pickle.dump(codebook_mappings_per_sample_dict, f)

        codebook_weights = model.vq_vae._embedding.weight.data.clone()
        umap_e = umap.UMAP(random_state=9).fit_transform(codebook_weights.cpu())

        pca = PCA(n_components=30)
        pca_result = pca.fit_transform(codebook_weights.cpu())
        plt.figure()
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Projection')
        plt.savefig(os.path.join(figdir, f'pca_ce_{epoch}.png'))

        kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(codebook_weights.cpu())
        plt.figure()
        plt.scatter(umap_e[:, 0], umap_e[:, 1], c=kmeans_labels, s=0.1, cmap='Spectral')
        plt.savefig(os.path.join(figdir, f'umap_ce_{epoch}.png'))

        model.eval()
        with torch.no_grad():
            for sample in val_subset:
                patient_name = os.path.basename(sample)
                cell_centroids = torch.load(os.path.join(sample, 'all_detected_cells_tiled_256_res_tensor.pt')).unsqueeze(1).to(torch.uint8)
                if cell_centroids.is_sparse: cell_centroids = cell_centroids.to_dense()

                posenc_cell_centroids = posenc_coords.repeat(cell_centroids.shape[0], 1, 1, 1)
                posenc_cell_centroids[(cell_centroids == 0).repeat(1, posenc_cell_centroids.shape[1], 1, 1)] = 0.

                det_transcripts = torch.load(os.path.join(sample, 'all_detected_transcripts_tiled_256_res_tensor.pt'))
                if det_transcripts.is_sparse: det_transcripts = det_transcripts.to_dense()
                images = torch.cat((cell_centroids, det_transcripts), dim=1)


                images = images[torch.randperm(images.shape[0])]
                val_images = torch.split(images, split_size_or_sections=batch_size, dim=0)
                val_pe = torch.split(posenc_cell_centroids, split_size_or_sections=batch_size, dim=0)
                for idx, batch in enumerate(val_images):
                    batch = torch.cat((val_pe[idx], batch, ), dim=1).float().to(device)
                    eval_vq_loss, reconstructed, eval_perplexity, encodings = model(batch)
                    encodings = encodings[-1]
                    eval_batch_loss, loss_dict = mse_ssim_loss(reconstructed, batch, use_sum=False,
                                                    ssim_module=ssim_module, mse_weight=1, ssim_weight=1,)
                    eval_batch_loss += eval_vq_loss
                    wandb.log({'eval_batch_loss': eval_batch_loss, 'eval_vq_loss': eval_vq_loss, 'eval_perplexity': eval_perplexity})


        wandb.log({'epoch': epoch, 'learning_rate': sched.get_lr()[0]})
        wandb.log({'codebook_usage': wandb.Histogram(encodings.flatten().cpu())})


if __name__ == '__main__':
    train_hier_vq_vae(dataset='vizgen_oncology')
