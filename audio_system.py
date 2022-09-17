import torch

class AudioSystem(torch.nn.Module):

    def __init__(self, classifier: torch.nn.Module, transform, defender: torch.nn.Module=None):

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
    
    def forward(self, x, defend=True):
        
        if defend == True and self.defender is not None:
            output = self.defender(x)
        else: 
            output = x
        
        if self.transform is not None: 
            output = self.transform(output)
            
        output = self.classifier(output)

        return output