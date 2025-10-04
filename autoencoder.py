import torch
import torch.nn as nn
from model import Encoder, Decoder
# from quantize import VectorQuantizer
from taming.modules.vqvae.quantize import VectorQuantizer
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator


class VQAutoEncoder(nn.Module):
    """
        see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
        ____________________________________________
        Discretization bottleneck part of the VQ-VAE.
        Inputs:
        - n_e : number of embeddings
        - e_dim : dimension of embedding
        - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        _____________________________________________
    """
    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, H):
        super(VQAutoEncoder, self).__init__()
        self.encoder = Encoder(ch=H.ch,
                               out_ch=H.out_ch,
                               num_res_blocks=H.num_res_blocks,
                               attn_resolutions=H.attn_resolutions,
                               resolution=H.resolution,
                               z_channels=H.z_channels,
                               ch_mult=H.ch_mult,
                               double_z=H.double_z)
        self.decoder = Decoder(ch=H.ch,
                               out_ch=H.out_ch,
                               ch_mult=H.ch_mult,
                               num_res_blocks=H.num_res_blocks,
                               attn_resolutions=H.attn_resolutions,
                               in_channels=H.in_channels,
                               resolution=H.resolution,
                               z_channels=H.z_channels)
        self.quantize = VectorQuantizer(n_e=H.codebook_size, e_dim=H.emb_dim, beta=0.25,
                                        remap=None,
                                        sane_index_shape=False)

        self.quant_conv = torch.nn.Conv2d(H.z_channels, H.emb_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(H.emb_dim, H.z_channels, 1)

    def forward(self, x):
        x = self.encoder(x)
        h = self.quant_conv(x)
        quant, codebook_loss, quant_stats = self.quantize(h)

        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec, codebook_loss, quant_stats


class VQGAN(nn.Module):
    def __init__(self, H):
        super(VQGAN, self).__init__()
        self.ae = VQAutoEncoder(H)
        self.loss = VQLPIPSWithDiscriminator(disc_start=H.disc_start_step,
                                             disc_in_channels=H.disc_in_channels,
                                             disc_conditional=H.disc_conditional,
                                             disc_weight=H.disc_weight,
                                             codebook_weight=H.codebook_weight)

    def get_last_layer(self):
        return self.ae.decoder.conv_out.weight

    def train_iter(self, x, step, generator_step):
        stats = {}
        xrec, codebook_loss, quant_stats = self.ae(x)
        if generator_step:
            ae_loss, log_dict_ae = self.loss(codebook_loss, x, xrec, 0, step,
                                            last_layer=self.get_last_layer(), split="train")
            stats["ae_loss"] = ae_loss
            stats["codebook_loss"] = codebook_loss.item()
        else:
            # discriminator
            disc_loss, log_dict_disc = self.loss(codebook_loss, x, xrec, 1, step,
                                                last_layer=self.get_last_layer(), split="train")
            stats["disc_loss"] = disc_loss
        return xrec, stats

