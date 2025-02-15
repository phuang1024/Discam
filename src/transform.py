"""
Solve least squares transform between features of frames.
"""

import matplotlib.pyplot as plt
import torch
from tqdm import trange


def solve_transform(from_pts, to_pts, iters=20, lr=1e-3, gamma=0.85):
    """
    Solve SE(2) transform via gradient descent.

    from_pts, to_pts: Array of shape [N, 2].
        Positions of features.
    """
    ones = torch.ones((from_pts.shape[0], 1), dtype=torch.float64)
    from_pts = torch.cat((from_pts, ones), dim=1).unsqueeze(2)
    to_pts = torch.cat((to_pts, ones), dim=1).unsqueeze(2)

    trans = torch.eye(3, dtype=torch.float64, requires_grad=True)

    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam([trans], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma)

    #losses = []
    for _ in range(iters):
        optim.zero_grad()
        pred = torch.matmul(trans, from_pts)
        loss = criterion(pred, to_pts)
        loss.backward()
        optim.step()
        scheduler.step()

        # Reset bottom row of transform.
        with torch.no_grad():
            trans[2, :2] = 0
            trans[2, 2] = 1

        #losses.append(loss.item())

    #plt.plot(losses)
    #plt.show()

    return trans


def extract_translation_zoom(transform):
    """
    Extract translation and zoom from SE(2) transform.
    """
    translation = (transform[0, 2].item(), transform[1, 2].item())
    zoom = (transform[0, 0].item() + transform[1, 1].item()) / 2
    return translation, zoom
