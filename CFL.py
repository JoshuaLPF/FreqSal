import torch
import torch.nn as nn
import torch.fft


class CoFocusFrequencyLoss(nn.Module):

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(CoFocusFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, dim=(2, 3), norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)

        return freq

    def loss_formulation(self, recon_freq, real_freq, x_freq, y_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance

            x_a = torch.abs(x_freq)
            x_p = torch.angle(x_freq)
            x_a = x_a.mean()
            x_r = x_a * torch.cos(x_p)
            x_i = x_a * torch.sin(x_p)
            x_freq = torch.complex(x_r, x_i)
            x_freq = x_freq.to(torch.float32)

            y_a = torch.abs(y_freq)
            y_p = torch.angle(y_freq)
            y_a = y_a.mean()
            y_r = y_a * torch.cos(y_p)
            y_i = y_a * torch.sin(y_p)
            y_freq = torch.complex(y_r, y_i)
            y_freq = y_freq.to(torch.float32)

            matrix_tmp = ((x_freq - real_freq) ** 2) + ((y_freq - real_freq) ** 2) + ((recon_freq - real_freq) ** 2)
            # matrix_tmp = ((x_freq - y_freq) ** 2)
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, x_pred, y_pred, matrix=None, **kwargs):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        x_freq = self.tensor2freq(x_pred)
        y_freq = self.tensor2freq(y_pred)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
            x_freq = torch.mean(x_freq, 0, keepdim=True)
            y_freq = torch.mean(y_freq, 0, keepdim=True)

        # calculate cross-modal frequency loss
        return self.loss_formulation(pred_freq, target_freq, x_freq, y_freq, matrix) * self.loss_weight



if __name__ == '__main__':
    a = torch.randn((1,1,384,384))
    b = torch.randn((1,1,384,384))
    c = torch.randn((1,1,384,384))
    d = torch.randn((1,1,384,384))
    a = torch.Tensor(a).cuda()
    b = torch.Tensor(b).cuda()
    c = torch.Tensor(c).cuda()
    d = torch.Tensor(d).cuda()
    cfl = CoFocusFrequencyLoss().cuda()
    e = cfl(a, b, c, d)
    print(e)