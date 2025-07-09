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
      <img src="res/vanilla_training_reconstruction.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/vanilla_validation_reconstruction.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/vanilla_conv1_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
      <img src="res/vanilla_conv2_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/vanilla_conv3_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
      <img src="res/vanilla_conv4_outputs.gif" alt="VAE Decoder Reconstruction" style="width:200px;height:200px;">
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
        </td>
        <td align="center">
            Samples from Inference<br>$\mathbf{z} \sim \mathcal{N}(0, \boldsymbol{I})$
        </td>
        <td align="center">
            Samples from Inference<br>per-class
        </td>
    </tr>

    

  <tr>
    <td align="center">
      Vanilla VAE
    </td>
    <td align="center">
      <img src="res/vanilla_samples.png" alt="Samples Vanilla" style="width:200px;height:200px;">
    </td>
    <td align="center">
      <img src="res/vanilla_samples.png" alt="Loss Vanilla" style="width:200px;height:200px;">
    </td>
  </tr>
</table>

<table>
    <tr>
    <td align="center">
      Loss
    </td>
    <td align="center">
      Reconstruction Loss
    </td>
    <td align="center">
      KL Divergence
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="res/vanilla_loss.png" alt="Loss Vanilla" style="width:330px;height:200px;">
    </td>
    <td align="center">
      <img src="res/vanilla_reconstruction.png" alt="Reconstruction Vanilla" style="width:330px;height:200px;">
    </td>
    <td align="center">
      <img src="res/vanilla_kl_div.png" alt="KL Vanilla" style="width:330px;height:200px;">
    </td>
  </tr>
</table>

<div style="  display: flex; justify-content: center; align-items: center;">
</div>

## Sources
- Auto-Encoding Variational Bayes [Diederik P Kingma, Max Welling. 2013](https://arxiv.org/pdf/1312.6114)
- $\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, [Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner 2017](https://openreview.net/pdf?id=Sy2fzU9gl)
