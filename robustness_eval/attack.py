import torch
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss, CTCLoss

from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
import scipy.signal as ss

class PsychoacousticMasker:
    """
    Implements psychoacoustic model of Lin and Abdulla (2015) following Qin et al. (2019) simplifications.

    | Paper link: Lin and Abdulla (2015), https://www.springer.com/gp/book/9783319079738
    | Paper link: Qin et al. (2019), http://proceedings.mlr.press/v97/qin19a.html
    """

    def __init__(self, window_size: int = 2048, hop_size: int = 512, sample_rate: int = 16000) -> None:
        """
        Initialization.

        :param window_size: Length of the window. The number of STFT rows is `(window_size // 2 + 1)`.
        :param hop_size: Number of audio samples between adjacent STFT columns.
        :param sample_rate: Sampling frequency of audio inputs.
        """
        self._window_size = window_size
        self._hop_size = hop_size
        self._sample_rate = sample_rate

        # init some private properties for lazy loading
        self._fft_frequencies: Optional[np.ndarray] = None
        self._bark: Optional[np.ndarray] = None
        self._absolute_threshold_hearing: Optional[np.ndarray] = None

    def calculate_threshold_and_psd_maximum(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the global masking threshold for an audio input and also return its maximum power spectral density.

        This method is the main method to call in order to obtain global masking thresholds for an audio input. It also
        returns the maximum power spectral density (PSD) for each frame. Given an audio input, the following steps are
        performed:

        1. STFT analysis and sound pressure level normalization
        2. Identification and filtering of maskers
        3. Calculation of individual masking thresholds
        4. Calculation of global masking thresholds

        :param audio: Audio samples of shape `(length,)`.
        :return: Global masking thresholds of shape `(window_size // 2 + 1, frame_length)` and the PSD maximum for each
            frame of shape `(frame_length)`.
        """
        psd_matrix, psd_max = self.power_spectral_density(audio)
        threshold = np.zeros_like(psd_matrix)
        for frame in range(psd_matrix.shape[1]):
            # apply methods for finding and filtering maskers
            maskers, masker_idx = self.filter_maskers(*self.find_maskers(psd_matrix[:, frame]))
            # apply methods for calculating global threshold
            threshold[:, frame] = self.calculate_global_threshold(
                self.calculate_individual_threshold(maskers, masker_idx)
            )
        return threshold, psd_max

    @property
    def window_size(self) -> int:
        """
        :return: Window size of the masker.
        """
        return self._window_size

    @property
    def hop_size(self) -> int:
        """
        :return: Hop size of the masker.
        """
        return self._hop_size

    @property
    def sample_rate(self) -> int:
        """
        :return: Sample rate of the masker.
        """
        return self._sample_rate

    @property
    def fft_frequencies(self) -> np.ndarray:
        """
        :return: Discrete fourier transform sample frequencies.
        """
        if self._fft_frequencies is None:
            self._fft_frequencies = np.linspace(0, self.sample_rate / 2, self.window_size // 2 + 1)
        return self._fft_frequencies

    @property
    def bark(self) -> np.ndarray:
        """
        :return: Bark scale for discrete fourier transform sample frequencies.
        """
        if self._bark is None:
            self._bark = 13 * np.arctan(0.00076 * self.fft_frequencies) + 3.5 * np.arctan(
                np.square(self.fft_frequencies / 7500.0)
            )
        return self._bark

    @property
    def absolute_threshold_hearing(self) -> np.ndarray:
        """
        :return: Absolute threshold of hearing (ATH) for discrete fourier transform sample frequencies.
        """
        if self._absolute_threshold_hearing is None:
            # ATH applies only to frequency range 20Hz<=f<=20kHz
            # note: deviates from Qin et al. implementation by using the Hz range as valid domain
            valid_domain = np.logical_and(20 <= self.fft_frequencies, self.fft_frequencies <= 2e4)
            freq = self.fft_frequencies[valid_domain] * 0.001

            # outside valid ATH domain, set values to -np.inf
            # note: This ensures that every possible masker in the bins <=20Hz is valid. As a consequence, the global
            # masking threshold formula will always return a value different to np.inf
            self._absolute_threshold_hearing = np.ones(valid_domain.shape) * -np.inf

            self._absolute_threshold_hearing[valid_domain] = (
                3.64 * pow(freq, -0.8) - 6.5 * np.exp(-0.6 * np.square(freq - 3.3)) + 0.001 * pow(freq, 4) - 12
            )
        return self._absolute_threshold_hearing

    def power_spectral_density(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectral density matrix for an audio input.

        :param audio: Audio sample of shape `(length,)`.
        :return: PSD matrix of shape `(window_size // 2 + 1, frame_length)` and maximum vector of shape
        `(frame_length)`.
        """
        import librosa

        # compute short-time Fourier transform (STFT)
        audio_float = audio.astype(np.float32)
        stft_params = {
            "n_fft": self.window_size,
            "hop_length": self.hop_size,
            "win_length": self.window_size,
            "window": ss.get_window("hann", self.window_size, fftbins=True),
            "center": False,
        }
        stft_matrix = librosa.core.stft(audio_float, **stft_params)

        # compute power spectral density (PSD)
        with np.errstate(divide="ignore"):
            gain_factor = np.sqrt(8.0 / 3.0)
            psd_matrix = 20 * np.log10(np.abs(gain_factor * stft_matrix / self.window_size))
            psd_matrix = psd_matrix.clip(min=-200)

        # normalize PSD at 96dB
        psd_matrix_max = np.max(psd_matrix)
        psd_matrix_normalized = 96.0 - psd_matrix_max + psd_matrix

        return psd_matrix_normalized, psd_matrix_max

    @staticmethod
    def find_maskers(psd_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify maskers.

        Possible maskers are local PSD maxima. Following Qin et al., all maskers are treated as tonal. Thus neglecting
        the nontonal type.

        :param psd_vector: PSD vector of shape `(window_size // 2 + 1)`.
        :return: Possible PSD maskers and indices.
        """
        # identify maskers. For simplification it is assumed that all maskers are tonal (vs. nontonal).
        masker_idx = ss.argrelmax(psd_vector)[0]

        # smooth maskers with their direct neighbors
        psd_maskers = 10 * np.log10(np.sum([10 ** (psd_vector[masker_idx + i] / 10) for i in range(-1, 2)], axis=0))
        return psd_maskers, masker_idx

    def filter_maskers(self, maskers: np.ndarray, masker_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter maskers.

        First, discard all maskers that are below the absolute threshold of hearing. Second, reduce pairs of maskers
        that are within 0.5 bark distance of each other by keeping the larger masker.

        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Filtered PSD maskers and indices.
        """
        # filter on the absolute threshold of hearing
        # note: deviates from Qin et al. implementation by filtering first on ATH and only then on bark distance
        ath_condition = maskers > self.absolute_threshold_hearing[masker_idx]
        masker_idx = masker_idx[ath_condition]
        maskers = maskers[ath_condition]

        # filter on the bark distance
        bark_condition = np.ones(masker_idx.shape, dtype=bool)
        i_prev = 0
        for i in range(1, len(masker_idx)):
            # find pairs of maskers that are within 0.5 bark distance of each other
            if self.bark[i] - self.bark[i_prev] < 0.5:
                # discard the smaller masker
                i_todelete, i_prev = (i_prev, i_prev + 1) if maskers[i_prev] < maskers[i] else (i, i_prev)
                bark_condition[i_todelete] = False
            else:
                i_prev = i
        masker_idx = masker_idx[bark_condition]
        maskers = maskers[bark_condition]

        return maskers, masker_idx

    def calculate_individual_threshold(self, maskers: np.ndarray, masker_idx: np.ndarray) -> np.ndarray:
        """
        Calculate individual masking threshold with frequency denoted at bark scale.

        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Individual threshold vector of shape `(window_size // 2 + 1)`.
        """
        delta_shift = -6.025 - 0.275 * self.bark
        threshold = np.zeros(masker_idx.shape + self.bark.shape)
        # TODO reduce for loop
        for k, (masker_j, masker) in enumerate(zip(masker_idx, maskers)):
            # critical band rate of the masker
            z_j = self.bark[masker_j]
            # distance maskees to masker in bark
            delta_z = self.bark - z_j
            # define two-slope spread function:
            #   if delta_z <= 0, spread_function = 27*delta_z
            #   if delta_z > 0, spread_function = [-27+0.37*max(PSD_masker-40,0]*delta_z
            spread_function = 27 * delta_z
            spread_function[delta_z > 0] = (-27 + 0.37 * max(masker - 40, 0)) * delta_z[delta_z > 0]

            # calculate threshold
            threshold[k, :] = masker + delta_shift[masker_j] + spread_function
        return threshold

    def calculate_global_threshold(self, individual_threshold):
        """
        Calculate global masking threshold.

        :param individual_threshold: Individual masking threshold vector.
        :return: Global threshold vector of shape `(window_size // 2 + 1)`.
        """
        # note: deviates from Qin et al. implementation by taking the log of the summation, which they do for numerical
        #       stability of the stage 2 optimization. We stabilize the optimization in the loss itself.
        with np.errstate(divide="ignore"):
            return 10 * np.log10(
                np.sum(10 ** (individual_threshold / 10), axis=0) + 10 ** (self.absolute_threshold_hearing / 10)
            )

class AudioAttack():
    '''
        Qin & CW WhiteBox attack
    '''
    def __init__(
        self,
        model: torch.nn.Module,
        transform = None, 
        defender = None, 
        masker: "PsychoacousticMasker" = None,
        criterion: "_Loss" = CrossEntropyLoss(),  
        eps: float = 2000.0,
        learning_rate_1: float = 100.0,
        max_iter_1: int = 1000,
        alpha: float = 0.05,
        learning_rate_2: float = 1.0,
        max_iter_2: int = 4000,
        loss_theta_min: float = 0.05,
        decrease_factor_eps: float = 0.8,
        num_iter_decrease_eps: int = 10,
        increase_factor_alpha: float = 1.2,
        num_iter_increase_alpha: int = 20,
        decrease_factor_alpha: float = 0.8,
        num_iter_decrease_alpha: int = 50,

        ) -> None:

        self.model = model
        self.transform = transform
        self.defender = defender
        self.masker = masker
        self.criterion = criterion

        self.eps = eps
        self.learning_rate_1 = learning_rate_1
        self.max_iter_1 = max_iter_1
        self.alpha = alpha
        self.learning_rate_2 = learning_rate_2
        self.max_iter_2 = max_iter_2

        self._targeted = True
        self.loss_theta_min = loss_theta_min
        self.decrease_factor_eps = decrease_factor_eps
        self.num_iter_decrease_eps = num_iter_decrease_eps
        self.increase_factor_alpha = increase_factor_alpha
        self.num_iter_increase_alpha = num_iter_increase_alpha
        self.decrease_factor_alpha = decrease_factor_alpha
        self.num_iter_decrease_alpha = num_iter_decrease_alpha
        
        self.scale_factor = 2**-15
    
    def generate(self, x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray], targeted: bool=True):
        
        self._targeted = targeted

        '''convert np.array to torch.tensor'''
        if isinstance(type(x), np.ndarray): 
            x = torch.from_numpy(x)
        if isinstance(type(y), np.ndarray): 
            y = torch.from_numpy(y)
        
        x_adv, success_count = self.stage_1(x, y)

        if self.max_iter_2 > 0:
            x_adv = self.stage_2(x_adv, y)

        return x_adv, success_count

    def stage_1(self, x: torch.Tensor, y: torch.Tensor):
        
        '''
            x: waveform tensor
            y: target
        '''
        if x.dtype == torch.float32:
            eps = self.scale_factor * self.eps
            lr = self.scale_factor * self.learning_rate_1
        else: 
            eps = self.eps
            lr = self.learning_rate_1

        batch_size = x.shape[0]
        x_adv = [None] * batch_size
        delta = torch.zeros_like(x, requires_grad=True)
        epsilon = [eps] * batch_size
        
        # torch.autograd.set_detect_anomaly(True)

        for i in range(1, self.max_iter_1 + 1):
            
            # with torch.autograd.detect_anomaly():
                
            '''update perturbed inputs'''
            x_in = x + delta

            if self.defender is not None: 
                x_in = self.defender(x_in)
            if self.transform is not None:
                x_in = self.transform(x_in)

            y_adv = self.model(x_in)
            loss = self.criterion(y_adv, y)
            
            loss.backward()

            x_pert = x + delta

            if i % self.num_iter_decrease_eps == 0:

                prediction = y_adv.max(1, keepdim=True)[1]

                if self._targeted: 
                    for j in range(batch_size):
                        if prediction[j] == y[j]:
                            # decrease max norm bound epsilon
                            perturbation_norm = torch.max(torch.abs(delta.data[j]))
                            if epsilon[j] > perturbation_norm:
                                epsilon[j] = perturbation_norm
                            epsilon[j] *= self.decrease_factor_eps
                            # save current best adversarial example
                            x_adv[j] = x_pert[j]
                else:
                    for j in range(batch_size):
                        if prediction[j] != y[j]:
                            # decrease max norm bound epsilon
                            perturbation_norm = torch.max(torch.abs(delta.data[j]))
                            if epsilon[j] > perturbation_norm:
                                epsilon[j] = perturbation_norm
                            epsilon[j] *= self.decrease_factor_eps
                            # save current best adversarial example
                            x_adv[j] = x_pert[j]

            '''update perturbations'''
            if self._targeted:
                delta.data = delta.data - lr * delta.grad.data.sign()
            else:
                delta.data = delta.data + lr * delta.grad.data.sign()
            delta.data = torch.cat([torch.clamp(torch.unsqueeze(p, 1), -e, e) for p, e in zip(delta.data, epsilon)], dim=0)
            delta.grad.zero_()

            # x_pert = x + delta

            # if i % self.num_iter_decrease_eps == 0:

            #     if self.defender is None: 
            #         prediction = self.model(self.transform(x_pert)).max(1, keepdim=True)[1]
            #     else:
            #         prediction = self.model(self.transform(self.defender(x_pert))).max(1, keepdim=True)[1]

            #     if self._targeted: 
            #         for j in range(batch_size):
            #             if prediction[j] == y[j]:
            #                 # decrease max norm bound epsilon
            #                 perturbation_norm = torch.max(torch.abs(delta.data[j]))
            #                 if epsilon[j] > perturbation_norm:
            #                     epsilon[j] = perturbation_norm
            #                 epsilon[j] *= self.decrease_factor_eps
            #                 # save current best adversarial example
            #                 x_adv[j] = x_pert[j]
            #     else:
            #         for j in range(batch_size):
            #             if prediction[j] != y[j]:
            #                 # decrease max norm bound epsilon
            #                 perturbation_norm = torch.max(torch.abs(delta.data[j]))
            #                 if epsilon[j] > perturbation_norm:
            #                     epsilon[j] = perturbation_norm
            #                 epsilon[j] *= self.decrease_factor_eps
            #                 # save current best adversarial example
            #                 x_adv[j] = x_pert[j]
        
        x_pert = x + delta
        success_count = batch_size

        ''' return perturbed x if no adversarial example found '''
        for j in range(batch_size):
            if x_adv[j] is None:
                print("Adversarial attack stage 1 for x_{} was not successful".format(j))
                x_adv[j] = x_pert[j]
                success_count = success_count - 1 

        x_adv = torch.unsqueeze(torch.cat(x_adv, dim=0), 1)

        return x_adv, success_count
    
    def stage_2(self, x: torch.Tensor, y: torch.Tensor=None):

        return x


from utils import MarginalLoss
class LinfSPSA():

    '''
        SPSA Attack (Uesato et al. 2018).
        Based on: https://arxiv.org/abs/1802.05666
        :param predict: predict function (single argument: input).
        :param eps: the L_inf budget of the attack.
        :param delta: scaling parameter of SPSA.
        :param lr: the learning rate of the `Adam` optimizer.
        :param nb_iter: number of iterations of the attack.
        :param nb_sample: number of samples for SPSA gradient approximation.
        :param max_batch_size: maximum batch size to be evaluated at once.
        :param targeted: [description]
        :param loss_fn: loss function (dual arguments: output, target).
        :param clip_min: upper bound of image values.
        :param clip_max: lower bound of image values.
    '''

    def __init__(self,
                model: torch.nn.Module,
                transform = None, 
                defender = None, 
                criterion: "_Loss" = MarginalLoss(),  
                eps: float = 2000.0,
                learning_rate: float = 100.0,
                max_iter: int = 1000,
                delta: float = 0.01,
                ) -> None:
        
        self.model = model
        self.transform = transform
        self.defender = defender
        self.criterion = criterion

        self.eps = eps
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.delta = delta

        self._targeted = True

        self.scale_factor = 2**-15
    
     
    def generate(self, 
                x: Union[torch.Tensor, np.ndarray], 
                y: Union[torch.Tensor, np.ndarray], 
                targeted: bool=True, 
                num_samples: int=512):
        
        self._targeted = targeted

        '''convert np.array to torch.tensor'''
        if isinstance(type(x), np.ndarray): 
            x = torch.from_numpy(x)
        if isinstance(type(y), np.ndarray): 
            y = torch.from_numpy(y)
        
        x_adv = self.spsa_perturb(x, y, num_samples)

        return x_adv

    def spsa_perturb(self, x: torch.Tensor, y: torch.Tensor, num_samples: int=64):

        dx = torch.zeros_like(x)
        dx.grad = torch.zeros_like(dx)

        if x.dtype == torch.float32:
            eps = self.scale_factor * self.eps
            lr = self.scale_factor * self.learning_rate
        else: 
            eps = self.eps
            lr = self.learning_rate

        optimizer = torch.optim.Adam([dx], lr=lr)

        for i in range(self.max_iter):

            optimizer.zero_grad()
            if self._targeted:
                dx.grad = self.spsa_grad(x, y, num_samples)
            else: 
                dx.grad = -self.spsa_grad(x, y, num_samples)
            optimizer.step()
            dx = dx.clamp(-eps, eps)

        x_adv = x + dx
        return x_adv

    @torch.no_grad()
    def predict(self, x: Union[torch.Tensor, np.ndarray]):

        '''convert np.array to torch.tensor'''
        if isinstance(type(x), np.ndarray): 
            x = torch.from_numpy(x)

        if self.defender is not None:
            x = self.defender(x, grad_enable=False)
        if self.transform is not None:
            x = self.transform(x)
        y = self.model(x)

        return y
    
    @torch.no_grad()
    def spsa_grad(self, x: torch.Tensor, y: torch.Tensor, num_samples: int=512, batch_size: int=16):      
        
        batches = [batch_size for _ in range(num_samples // batch_size)]
        if num_samples % batch_size:
            batches.append(num_samples % batch_size)

        grad = torch.zeros_like(x)

        for batch in batches:

            x_batch = x.repeat(batch, 1, 1)
            y_batch = y.repeat(batch)
            v = torch.empty_like(x_batch)
            v = v.bernoulli_().mul_(2.0).sub_(1.0)


            df_n = self.criterion(self.predict(x_batch + self.delta * v), y_batch) \
                - self.criterion(self.predict(x_batch - self.delta * v), y_batch)
            grad_n = df_n / (2. * self.delta * v)

            grad = grad + sum(grad_n.split(x.shape[0], dim=0))

        grad = grad / (num_samples * x.shape[0])

        return grad