import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from backbone.dfformer import cdfformer_m36, StarReLU
from timm.models.layers.helpers import to_2tuple
import numpy as np
from thop import profile
from thop import clever_format


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class edge_module(nn.Module):
    def __init__(self, dim,size):
        super(edge_module, self).__init__()
        self.phase_enhancement_t = nn.Sequential(
            nn.Conv2d(dim,dim,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(dim,dim,1,1,0))
        self.phase_enhancement_f = nn.Sequential(
            nn.Conv2d(dim,dim,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(dim,dim,1,1,0))
        self.complex_weight = nn.Parameter(torch.randn(size, size//2+1, dim, 2, dtype=torch.float32) * 0.02) # 14,8
        self.FFTRCAB_edge = FFTRCAB(dim)
        self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, kernel_size=3, stride=1, padding=1, groups=dim//4, bias=False),
                                       nn.BatchNorm2d(dim//4),
                                       nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))

    def forward(self, t, f):
        res_f = f
        t = t.to(torch.float32)
        f = f.to(torch.float32)

        # high-frequency filters
        weight = torch.view_as_complex(self.complex_weight).permute(2, 0, 1) # 56,29,64

        # phase enhancment for t
        t_fft = torch.fft.rfft2(t, dim=(2, 3), norm='ortho')
        mag_t = torch.abs(t_fft)
        pha_t = torch.angle(t_fft)
        pha_enh_t = self.phase_enhancement_t(pha_t)
        real_t = mag_t * torch.cos(pha_enh_t)
        imag_t = mag_t * torch.sin(pha_enh_t)
        pha_enhance_t = torch.complex(real_t, imag_t)

        # phase enhancment for f
        f_fft = torch.fft.rfft2(f, dim=(2, 3), norm='ortho')
        mag_f = torch.abs(f_fft)
        pha_f = torch.angle(f_fft)
        pha_enh_f = self.phase_enhancement_f(pha_f)
        real_f = mag_f * torch.cos(pha_enh_f)
        imag_f = mag_f * torch.sin(pha_enh_f)
        pha_enhance_f = torch.complex(real_f, imag_f)

        pha_enh = pha_enhance_t + pha_enhance_f
        fft_high = pha_enh * weight
        ifft_HighFre = torch.fft.irfft2(fft_high, dim=(2, 3), norm='ortho')

        High_fea = ifft_HighFre + res_f
        edge_fea_out = self.conv(self.FFTRCAB_edge(High_fea))
        return edge_fea_out


class Fusion(nn.Module):
    def __init__(self, dim, size, reweight_expansion_ratio=.25, num_filters=8): # , bias=False, sparsity_threshold=0.01, **kwargs,
        super(Fusion, self).__init__()
        self.size = size
        self.filter_size = size//2 + 1
        self.dim = dim
        self.conv_r = nn.Conv2d(dim,dim,1,1,0)
        self.act_r = StarReLU()
        self.conv_t = nn.Conv2d(dim,dim,1,1,0)
        self.act_t = StarReLU()
        self.bn_r = nn.BatchNorm2d(dim)
        self.bn_t = nn.BatchNorm2d(dim)
        self.num_filters = num_filters
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * dim)
        self.complex_weights = nn.Parameter(torch.randn(self.size, self.filter_size, num_filters, 2, dtype=torch.float32) * 0.02)
        self.rc_aEnhance = nn.Sequential(
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))
        self.rc_pEnhance = nn.Sequential(
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))
        self.tc_aEnhance = nn.Sequential(
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))
        self.tc_pEnhance = nn.Sequential(
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))
        self.DwconvFFN = nn.Sequential(nn.BatchNorm2d(dim),
                                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))

        self.conv_phase_r = nn.Sequential(
            nn.Conv2d(dim,dim,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(dim,dim,1,1,0))
        self.conv_phase_t = nn.Sequential(
            nn.Conv2d(dim,dim,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(dim,dim,1,1,0))

    def forward(self, r, t):
        B, _, H, W = r.shape
        res_r = r
        res_t = t
        r = self.bn_r(r)
        t = self.bn_t(t)
        x = r + t
        routeing = self.reweight(x.mean(dim=(2, 3))).view(B, self.num_filters, -1).softmax(dim=1)

        r = self.conv_r(r)
        r = self.act_r(r)
        t = self.conv_t(t)
        t = self.act_t(t)
        r = r.to(torch.float32)
        t = t.to(torch.float32)

        r_fft = torch.fft.rfft2(r, dim=(2, 3), norm='ortho')
        t_fft = torch.fft.rfft2(t, dim=(2, 3), norm='ortho')
        rs_a = torch.abs(r_fft)
        rs_p = torch.angle(r_fft)
        ts_a = torch.abs(t_fft)
        ts_p = torch.angle(t_fft)

        rs_fft = torch.fft.rfft2(rs_a, dim=1, norm='ortho')
        ts_fft = torch.fft.rfft2(ts_a, dim=1, norm='ortho')
        rsc_a = torch.abs(rs_fft)
        rsc_p = torch.angle(rs_fft)
        tsc_a = torch.abs(ts_fft)
        tsc_p = torch.angle(ts_fft)
        rsc_aEnh = self.rc_aEnhance(rsc_a)
        rsc_pEnh = self.rc_pEnhance(rsc_p)
        tsc_aEnh = self.tc_aEnhance(tsc_a)
        tsc_pEnh = self.tc_pEnhance(tsc_p)
        rsc_r = rsc_aEnh * torch.cos(rsc_pEnh)
        rsc_i = rsc_aEnh * torch.sin(rsc_pEnh)
        rsc_comp = torch.complex(rsc_r, rsc_i)
        tsc_r = tsc_aEnh * torch.cos(tsc_pEnh)
        tsc_i = tsc_aEnh * torch.sin(tsc_pEnh)
        tsc_comp = torch.complex(tsc_r, tsc_i)
        rc = torch.fft.irfft2(rsc_comp, dim=1, norm='ortho')
        tc = torch.fft.irfft2(tsc_comp, dim=1, norm='ortho')

        rsc = rc * rs_a
        tsc = tc * ts_a
        rs_p = self.conv_phase_r(rs_p)
        rs_r = rsc * torch.cos(rs_p)
        rs_i = rsc * torch.sin(rs_p)
        rs_comp = torch.complex(rs_r, rs_i)
        ts_p = self.conv_phase_t(ts_p)
        ts_r = tsc * torch.cos(ts_p)
        ts_i = tsc * torch.sin(ts_p)
        ts_comp = torch.complex(ts_r, ts_i)
        rt_s = rs_comp + ts_comp
        weight = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weights = torch.einsum('bfc,hwf->bchw', routeing, weight)
        weights = weights.view(-1, self.dim, self.size, self.filter_size)

        rt_sEnh = rt_s * weights
        rt = torch.fft.irfft2(rt_sEnh, dim=(2, 3), norm='ortho')
        out_1 = rt + res_r + res_t
        out = self.DwconvFFN(out_1) + out_1
        return out


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class FFTRCAB(nn.Module):
    def __init__(self, dim):
        super(FFTRCAB, self).__init__()

        self.CBG3x3 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.xc_aEnhance = nn.Sequential(
            nn.Conv2d(dim, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))
        self.xc_pEnhance = nn.Sequential(
            nn.Conv2d(dim, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))

    def forward(self, x):
        x_conv = self.CBG3x3(x)
        x_conv = x_conv.to(torch.float32)
        x_pool = self.avg_pool(x_conv)
        xc_a = self.xc_aEnhance(x_pool)
        xc_p = self.xc_pEnhance(x_pool)
        x_fft = torch.fft.rfft2(x_pool, dim=1, norm='ortho')
        x_a = torch.abs(x_fft)
        x_p = torch.angle(x_fft)
        xa_enh = x_a * xc_a
        xp_enh = x_p * xc_p
        xa = xa_enh * torch.cos(xp_enh)
        xp = xa_enh * torch.sin(xp_enh)
        x_comp = torch.complex(xa, xp)
        xc = torch.fft.irfft2(x_comp, dim=1, norm='ortho')
        x_out = x_conv * xc
        return x_out + x


class FreqSal(nn.Module):
    def __init__(self):
        super(FreqSal, self).__init__()

        # encoder
        self.rgb_feature = cdfformer_m36()
        self.depth_feature = cdfformer_m36()

        # fusion part
        self.fusion4 = Fusion(576, 12)
        self.fusion3 = Fusion(384, 24)
        self.fusion2 = Fusion(192, 48)
        self.fusion1 = Fusion(96, 96)

        # decoder
        self.upsample1 = nn.Sequential(nn.Conv2d(528, 132, kernel_size=3, stride=1, padding=1, groups=132, bias=False),
                                       nn.BatchNorm2d(132),
                                       nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample2 = nn.Sequential(nn.Conv2d(468, 156, kernel_size=3, stride=1, padding=1, groups=156, bias=False),
                                       nn.BatchNorm2d(156),
                                       nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample3 = nn.Sequential(nn.Conv2d(528, 132, kernel_size=3, stride=1, padding=1, groups=132, bias=False),
                                       nn.BatchNorm2d(132),
                                       nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample4 = nn.Sequential(nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1, groups=144, bias=False),
                                       nn.BatchNorm2d(144),
                                       nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.FFTRCAB4 = FFTRCAB(576)
        self.FFTRCAB34 = FFTRCAB(528)
        self.FFTRCAB234 = FFTRCAB(468)
        self.FFTRCAB1234 = FFTRCAB(528)

        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.edge1 = edge_module(96, 96)
        self.edge2 = edge_module(192, 48)
        self.up2_96 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.Dwconv72_72 = nn.Sequential(nn.Conv2d(72, 72, kernel_size=3, stride=1, padding=1, groups=72, bias=False),
                                       nn.BatchNorm2d(72),
                                       nn.ReLU())
        self.Up_Edge = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2, ),
                                     nn.Conv2d(72, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False))

        self.S4 = nn.Conv2d(144, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(132, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(156, 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(132, 1, 3, stride=1, padding=1)
        self.up4_384 = nn.Upsample(384)
        self.up3_384 = nn.Upsample(384)
        self.up2_384 = nn.Upsample(384)
        self.up1_384 = nn.Upsample(384)

        self.Dwconv204_51 = nn.Sequential(nn.Conv2d(204, 51, kernel_size=3, stride=1, padding=1, groups=51, bias=False),
                                       nn.BatchNorm2d(51),
                                       nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.Dwconv51_1 = nn.Sequential(nn.Conv2d(51, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
                                       nn.BatchNorm2d(1),
                                       nn.ReLU())
        self.Dwconv96_1_r = nn.Sequential(nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
                                       nn.UpsamplingBilinear2d(scale_factor=4, ))
        self.Dwconv96_1_t = nn.Sequential(nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
                                       nn.UpsamplingBilinear2d(scale_factor=4, ))

    def forward(self, rgb, t):
        # encoder
        r = self.rgb_feature(rgb)
        t = self.depth_feature(t)
        r1 = r[0]
        r2 = r[1]
        r3 = r[2]
        r4 = r[3]
        t1 = t[0]
        t2 = t[1]
        t3 = t[2]
        t4 = t[3]

        # fusion part
        f1 = self.fusion1(r1, t1)
        f2 = self.fusion2(r2, t2)
        f3 = self.fusion3(r3, t3)
        f4 = self.fusion4(r4, t4)

        # Pyramid decoder
        F4 = self.FFTRCAB4(f4)
        F4 = self.upsample4(F4)
        F34 = torch.cat((f3, F4), dim=1)
        F34 = self.FFTRCAB34(F34)
        F34 = self.upsample3(F34)
        F234 = torch.cat((self.up2(F4), F34, f2), dim=1)
        F234 = self.FFTRCAB234(F234)
        F234 = self.upsample2(F234)
        F1234 = torch.cat((self.up4(F4), self.up2(F34), F234, f1), dim=1)
        F1234 = self.FFTRCAB1234(F1234)
        F1234 = self.upsample1(F1234)

        # multi-level loss
        s4 = self.up4_384(self.S4(F4))
        s3 = self.up3_384(self.S3(F34))
        s2 = self.up2_384(self.S2(F234))
        s1 = self.up1_384(self.S1(F1234))

        # edge part
        edge_feature1 = self.edge1(t1, f1)
        edge_feature2 = self.edge2(t2, f2)
        edge_feature2 = self.up2_96(edge_feature2)
        edge_feature = torch.cat((edge_feature1, edge_feature2), dim=1)
        edge_feature = self.Dwconv72_72(edge_feature)
        up_edge = self.Up_Edge(edge_feature)

        # saliency head
        out_cat = torch.cat((F1234, edge_feature), dim=1)
        out = self.Dwconv204_51(out_cat)
        out = self.Dwconv51_1(out)

        wr = self.Dwconv96_1_r(r1)
        wt = self.Dwconv96_1_t(t1)

        return out, up_edge, s1, s2, s3, s4, wr, wt

    def load_pre(self, pre_model):
        self.rgb_feature.load_state_dict(torch.load(pre_model),strict=False)
        print(f"RGB loading pre_model ${pre_model}")
        self.depth_feature.load_state_dict(torch.load(pre_model), strict=False)
        print(f"Depth loading pre_model ${pre_model}")


if __name__ == '__main__':
    a = np.random.random((1,3,384,384))
    b = np.random.random((1,3,384,384))
    c = torch.Tensor(a).cuda()
    d = torch.Tensor(b).cuda()
    swinNet = FreqSal().cuda()
    e = swinNet(c,d)
    flops, params = profile(swinNet, inputs=(c, d))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops, params)