import torch

class AudioSystem(torch.nn.Module):

    def __init__(self, classifier, transform, defender):

        super().__init__()

        '''
            the whole audio system
            defender: audio -> audio, diffusion model for purification
            transform: audio -> spectrogram
            classifier: spectrogram -> prediction probability distribution
        '''

        self.classifier = classifier
        self.transform = transform
        self.defender = defender
    
    def forward(self, x):

        output = self.defender(x)
        output = self.transform(output)
        output = self.classifier(output)

        return output