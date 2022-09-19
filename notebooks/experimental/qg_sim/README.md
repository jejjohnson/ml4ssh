# Notebooks

In these notebooks, we have different realizations of using NerFs to train on QG simulations that are regularized by PINNs.


---
## Notebook 1: `1_nerf_qg_full.ipynb`

The full QG equation is given by:

$$
\begin{aligned}
\partial_t q + \det \boldsymbol{J}(q, \psi) &= 0
\end{aligned}
$$

where:

* $q=\nabla^2 \psi$
* $\det \boldsymbol{J}(q, \psi)=\partial_x q\partial_y\psi - \partial_y q\partial_x\psi$.

We are interested in finding some NerF method that can take in the spatial-temporal coordinates, $\mathbf{x}_\phi$, and output a vector corresponding to the PV and stream function, $\psi$, i.e. $\mathbf{y}_\text{obs}$.

$$
\mathbf{y}_\text{obs} = \boldsymbol{f_\theta}(\mathbf{x}_\phi) + \epsilon, \hspace{5mm}\epsilon \sim \mathcal{N}(0, \sigma^2)
$$

We use a SIREN network which is a fully connected neural network with the $sin$ activation function.

* **Data Inputs**: `256x256x11`
* **Data Ouputs**: `2`

---
## Notebook II: `2_nerf_qg_stream.ipynb`  (**TODO**)


$$
\begin{aligned}
\partial_t \nabla^2 \psi + \det \boldsymbol{J}(\nabla^2\psi, \psi) &= 0 \\
\partial_t \nabla^2 \psi + \partial_x \nabla^2 \psi\partial_y\psi - \partial_y \nabla^2 \psi\partial_x\psi &= 0
\end{aligned}
$$

where:

* $\det \boldsymbol{J}(q, \psi)=\partial_x q\partial_y\psi - \partial_y q\partial_x\psi$.


---
## Notebook III: `3_nerf_qg_ssh.ipynb` (**TODO**)


$$
\begin{aligned}
\partial_t \nabla^2 \eta + \frac{g}{f}\det \boldsymbol{J}(\nabla^2\eta, \eta) &= 0
\end{aligned}
$$

where:

* $g$ is the reduced gravity constant
* $f$ is the Coriolis parameter
* $\det \boldsymbol{J}(\nabla^2\eta, \psi)=\partial_x \eta\partial_y\psi - \partial_y \eta\partial_x\psi$.
