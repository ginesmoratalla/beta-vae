# Variational Autoencoder (VAE)

## Training Evolution (MNIST)

<table>
  <tr>
    <td align="center">
    </td>
    <td align="center">
      Decoder Reconstruction - Training Set
    </td>
    <td align="center">
      Decoder Reconstruction - Validation Set
    </td>
    <td align="center">
      Convolutional Layers Output - Encoder
    </td>
    <td align="center">
      Convolutional Layers Output - Decoder
    </td>
  </tr>
  
  <tr>
    <td align="center">
      Vanilla VAE
    </td>
    <td align="center">
      <img src="res/MNIST/vanilla_training_reconstruction.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/vanilla_validation_reconstruction.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/vanilla_conv1_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
      <img src="res/MNIST/vanilla_conv2_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/vanilla_conv3_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
      <img src="res/MNIST/vanilla_conv4_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
  </tr>

  <tr>
    <td align="center">
      $\beta$-VAE (0.3 $\beta$ incremental)
    <td align="center">
      <img src="res/MNIST/beta_training_reconstruction.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/beta_validation_reconstruction.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/beta_conv1_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
      <img src="res/MNIST/beta_conv2_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/beta_conv3_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
      <img src="res/MNIST/beta_conv4_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
  </tr>

  <!--
  <tr>
    <td align="center">
      β-VAE
    </td>
  </tr>
  -->
</table>

<table>
  <tr>
      <td align="center">
      </td> <td align="center">
          Samples from Inference<br>$\mathbf{z} \sim \mathcal{N}(0, \boldsymbol{I})$
      </td>
      <td align="center">
          Samples from Inference (per-class)<br>$\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{class}}, \boldsymbol{\sigma}_{\text{class}})$
      </td>
  </tr>
  
  <tr>
    <td align="center">
      Vanilla VAE
    </td>
    <td align="center">
      <img src="res/MNIST/vanilla_samples.png" alt="Samples Vanilla" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/vanilla_samples_per_class.png" alt="Loss Vanilla" style="width:300px;height:300px;">
    </td>
  </tr>

  <tr>
    <td align="center">
      $\beta$-VAE (0.3 $\beta$ incremental)
    </td>
    <td align="center">
      <img src="res/MNIST/beta_samples.png" alt="Samples Vanilla" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/beta_samples_per_class.png" alt="Loss Vanilla" style="width:300px;height:300px;">
    </td>
  </tr>

  <tr>
    <td align="center">
      $\beta$-VAE (0.3 $\beta$ incremental +10 epochs)
    </td>
    <td align="center">
      <img src="res/MNIST/beta_1_third.png" alt="Samples Vanilla" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/beta_1_third_samples_per_class.png" alt="Loss Vanilla" style="width:300px;height:300px;">
    </td>
  </tr>
  
</table>

<table>
  <tr>
    <td align="center">
      Reconstruction Loss
    </td>
    <td align="center">
      KL Divergence
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="res/MNIST/reconstruction_loss.png" alt="Reconstruction Vanilla" style="width:330px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/kl_div.png" alt="KL Vanilla" style="width:330px;height:200px;">
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
        Image sampling transition $\beta$-VAE
    </td>
      <td align="center">
          3D Principal Component Analysis on $\mathbf{z}$ ($\mathbf{\mu_z}$)<br>$\mathbf{z}\in\mathbb{R}^{70}$
      </td>
  </tr>
  <tr>
    <td align="center">
      <img src="res/MNIST/number_transitions.gif" alt="Reconstruction Vanilla" style="width:330px;height:200px;">
    </td>
    <td align="center">
      <img src="res/MNIST/pca.gif" alt="PCA Vanilla" style="width:300px;height:300px;">
    </td>
  </tr>
</table>

<div style="  display: flex; justify-content: center; align-items: center;">
</div>

## Sampled Attribute Transitions (CelebA)


<table>
  <tr>
      <td align="center">
      </td>
      <td align="center">
            Decoder Reconstruction - Training Set
      </td>
      <td align="center">
            Decoder Reconstruction - Validation Set
      </td>
  </tr>
  <tr>
    <td align="center">
      $\beta$-VAE<br>(0.3 $\beta$ incremental)<br>$\mathbf{z}\in\mathbb{R}^{40}$
    </td>
    <td align="center">
      <img src="res/CelebA/training_40_dim.gif" alt="Samples Vanilla" style="width:250px;height:250px;">
    </td>
    <td align="center">
      <img src="res/CelebA/validation_40.gif" alt="Loss Vanilla" style="width:250px;height:250px;">
    </td>
  </tr>

  <tr>
    <td align="center">
      $\beta$-VAE<br>(0.3 $\beta$ incremental)<br>$\mathbf{z}\in\mathbb{R}^{200}$
    </td>
    <td align="center">
      <img src="res/CelebA/training_200_dim.gif" alt="Samples Vanilla" style="width:250px;height:250px;">
    </td>
    <td align="center">
      <img src="res/CelebA/validation_200_dim.gif" alt="Loss Vanilla" style="width:250px;height:250px;">
    </td>
  </tr>
</table>


<table>
  <tr>
    <td align="center">
        Image sampling transition (Bald attribute)
    </td>
      <td align="center">
        Image sampling transition (Gender attribute)
      </td>
  </tr>
  <tr>
    <td align="center">
      <img src="res/CelebA/bald-transition.png" alt="Reconstruction Vanilla" style="width:330px;height:300px;">
    </td>
    <td align="center">
      <img src="res/CelebA/gender-transition.png" alt="Reconstruction Vanilla" style="width:330px;height:300px;">
    </td>
  </tr>

  <tr>
    <td align="center">
            Image sampling transition (Hair Colour attribute)
    </td>
    <td align="center">
            Image sampling transition (Pale Skin attribute)
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="res/CelebA/hair-transition.png" alt="Reconstruction Vanilla" style="width:330px;height:300px;">
    </td>
    <td align="center">
      <img src="res/CelebA/pale_skin-transition.png" alt="Reconstruction Vanilla" style="width:330px;height:300px;">
    </td>
  </tr>

</table>

## Sources
- Auto-Encoding Variational Bayes [Diederik P Kingma, Max Welling. 2013](https://arxiv.org/pdf/1312.6114)
- $\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, [Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner 2017](https://openreview.net/pdf?id=Sy2fzU9gl)
