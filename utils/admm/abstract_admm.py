import torch
from abc import ABC, abstractmethod

__all__ = ['AbstractAdmmScheduler']


class AbstractAdmmScheduler(ABC):
    def __init__(self, model):
        self.model = model
        self.Z = ()
        self.U = ()
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) != 1:  # len(BN.weight) = 1
                self.Z += (param.detach().clone(),)
                self.U += (torch.zeros_like(param),)
        super().__init__()

    def update_per_epoch(self):
        X = self.get_current_X()
        self.update_Z(X)
        self.update_U(X)
        return self.print_convergence(X)

    @abstractmethod
    def custom_Z_regulation(self, z):
        """
        Example (Level Pruning):
            pcen = np.percentile(abs(z), 100 * self.percent[idx])
            under_threshold = abs(z) < pcen
            z.data[under_threshold] = 0
        """
        raise NotImplementedError

    def print_convergence(self, X):
        idx = 0
        print("normalized norm of (weight - projection)")
        ave_convergence = 0
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) != 1:
                x, z = X[idx], self.Z[idx]
                cg = (x - z).norm().item() / x.norm().item()
                print("({}): {:.4f}".format(name, cg))
                idx += 1
                ave_convergence += cg
        return ave_convergence / idx

    def get_current_X(self):
        X = ()
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) != 1:
                X += (param.detach().clone(),)
        return X

    def update_Z(self, X):  # todo: custom
        new_Z = ()
        idx = 0
        for x, u in zip(X, self.U):
            z = x + u
            z = self.custom_Z_regulation(z)
            new_Z += (z,)
            idx += 1
        self.Z = new_Z

    def update_U(self, X):
        new_U = ()
        for u, x, z in zip(self.U, X, self.Z):
            new_u = u + x - z
            new_U += (new_u,)
        self.U = new_U
